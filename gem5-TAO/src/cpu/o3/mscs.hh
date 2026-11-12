/*
 * Multi-stage CPI stack (MSCS) accounting, as specified in the paper:
 *
 *   Every stage offers W_min slots per cycle.
 *   (R1) Base slots: each correct-path instruction mu_i occupies one Base
 *        slot per stage, filling the slots of the cycle in which the stage
 *        processes it, in processing order; any beyond W_min spill into the
 *        next cycle (Eq. 2).  Wrong-path work never occupies a Base slot.
 *   (R2) Lost slots: every other slot is charged to its root cause.
 *        BP  if the stage holds a correct-path instruction it cannot advance
 *            (B_c^s), at every stage;
 *        at Decode (no predecessor), from the current cycle's conditions:
 *        SP  if a wrong-path instruction is processed (X_c);
 *        FP  if not X_c and (I_c or not M_c);
 *        SP  if not X_c and not I_c and M_c                       (Eq. 3)
 *        at a later stage s that is not blocked, the slot inherits the
 *        class of the slot at the same position of its predecessor s-
 *        d_s cycles earlier                                       (Eq. 4)
 *
 * Whether an instruction is on the correct path, whether a branch was
 * mispredicted and whether an I-cache access missed are only known in
 * hindsight, so the accountant records raw events during simulation and
 * classifies all slots when the simulator exits (window = whole run of this
 * CPU, which in the data-generation flow is the measured O3 window).
 *
 * Implementation notes (details the paper leaves open):
 *  - Stage mapping: Decode = Decode::decodeInsts, Dispatch = rename's send
 *    to IEW (where ROB/IQ/LSQ/register stalls block in gem5), Issue = final
 *    issue from the IQ (or IEW dispatch for IQ-bypassing NOPs), Commit.
 *    Decode's predecessor for B_c is Fetch (send to decode).
 *  - B_c^s (Table 1): the stage's structural back-end condition holds in
 *    cycle c and the instruction it is stuck on is on the correct path:
 *      Decode   DDQ full (Rename stalls Decode); stuck = oldest in Decode
 *      Dispatch ROB or RS (IQ) full; stuck = oldest in Rename
 *      Issue    RS holds a non-ready instruction; stuck = oldest non-ready
 *      Commit   ROB head not done; stuck = ROB head
 *    Conditions are sampled at the end of each stage's tick; cycles in
 *    which the CPU does not tick (idle) keep the last sampled state.
 *  - d_s defaults to the minimum observed latency from s- to s over the
 *    window.
 *  - A later stage that is not blocked but whose referenced slot (Eq. 4) is
 *    a Base slot (stuck for a reason outside Table 1, e.g. physical
 *    registers full) is charged to BP and counted as fallback_slots.
 *  - Stage s's window of C cycles is offset by d_Dispatch+...+d_s from
 *    Decode's, so Eq. 4 maps the window of s one-to-one onto that of s-.
 *    C is the smallest length that holds all n Base slots at every stage,
 *    so every stage has exactly n Base slots and CPI = C / n.
 *  - Instructions still in flight when the simulation ends (not squashed,
 *    not committed) are treated as correct-path for B/X/M but are outside
 *    the window, so they never take Base slots.
 *  - I_c: fetch was waiting on an ITLB miss, an I-cache miss or an I-cache
 *    retry for the line that delivers the next correct-path instruction.
 *  - M_c: covers [decode cycle of a mispredicted branch, decode cycle of
 *    the next correct-path instruction); mispredicts detected at Decode and
 *    at IEW both count.
 *
 * Trace line (28 fields after "START"), same layout as the previous format:
 *   Decode   [Base, FP, BP, SP]
 *   Dispatch [Base, FP, BP0, BP1, BP2, SP]
 *   IEW      [Base, FP, BP0, BP1, BP2, SP]
 *   Issue    [Base, FP, BP0, BP1, BP2, SP]
 *   Commit   [Base, FP, BP0, BP1, BP2, SP]
 * BP0 + BP1 + BP2 = BP of that stage.  The breakdown (BPSub) is taken from
 * the state of the stage in the cycle the BP slot is charged:
 *   Dispatch ROB/IQ full (Table 1) | physical registers full | serialize
 *   IEW      other                 | IQ full                 | LQ/SQ full
 *   Issue    stuck on a non-load   | stuck on a load         | FU busy
 *   Commit   head long-latency op  | head memory op          | other
 * BP inherited from the predecessor (Eq. 4) carries no stage-local cause
 * and goes to BP0, as do slots with no recorded cause.
 * IEW (dispatch into IQ/LSQ) is not one of the four MSCS stages.  It is
 * computed with the same rules (predecessor Dispatch, B_c = IQ or LSQ full
 * for the instruction at the head of IEW) but does not take part in the
 * Dispatch -> Issue chain or in the choice of C.
 */

#ifndef __CPU_O3_MSCS_HH__
#define __CPU_O3_MSCS_HH__

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "base/types.hh"
#include "cpu/inst_seq.hh"

namespace gem5
{

namespace o3
{

class MSCSAccountant
{
  public:
    /** Stamp indices. FETCH is only the predecessor of Decode. */
    enum Stamp { FETCH = 0, DEC = 1, DISP = 2, ISS = 3, COM = 4,
                 IEWD = 5, NumStamps = 6 };
    /** Stage rows: Dec, Disp, Iss, Com (MSCS) and IEW. */
    static constexpr int NumRows = 5;
    static constexpr int IEW_ROW = 4;
    static constexpr int NumSub = 3;
    /** Slot classes (output order). */
    enum Class : uint8_t { BASE = 0, FP = 1, BP = 2, SP = 3, NumClasses = 4 };
    /** BP breakdown, in trace sub-field order (see top of file). */
    enum BPSub : uint8_t {
        DISP_ROB_IQ = 0, DISP_REG = 1, DISP_SERIALIZE = 2,
        IEW_OTHER = 0, IEW_IQ = 1, IEW_LSQ = 2,
        ISS_DEPEND = 0, ISS_MEM = 1, ISS_FU = 2,
        COM_FU = 0, COM_MEM = 1, COM_DEPEND = 2
    };

    struct InstRec
    {
        InstSeqNum seq = 0;
        int64_t t[NumStamps] = {-1, -1, -1, -1, -1, -1};
        bool mispred = false;
    };

    struct Result
    {
        bool valid = false;
        /** Row (0..4 = Dec, Disp, Iss, Com, IEW) x class proportions. */
        double v[NumRows][NumClasses] = {};
        /** BP breakdown per row; sums to v[row][BP]. */
        double sub[NumRows][NumSub] = {};
    };

    /**
     * @param w_min      slot width
     * @param delay_cfg  d_s for Dec, Disp, Iss, Com; < 0 means "use the
     *                   minimum latency observed in the window"
     */
    void init(int w_min, const int delay_cfg[4], FILE *trace,
              const std::string &diag_path);

    /** Called at the start of every CPU tick. */
    void tick(Cycles now);

    /** Decode handled (decoded or dropped) n instructions this cycle. */
    void decodeHandled(Cycles now, int n);

    /**
     * Table 1 back-end condition of stage (DEC..COM, IEWD) holds this cycle;
     * held_seq is the instruction the stage is stuck on, sub its BPSub.
     */
    void blocked(int stage, Cycles now, InstSeqNum held_seq,
                 uint8_t sub = 0);

    /**
     * Stage is stalled this cycle for a back-end reason outside Table 1
     * (used only for the BP breakdown of slots charged to BP).
     */
    void stallReason(int stage, Cycles now, uint8_t sub);

    /** Fetch stalled on an I-side miss from start to end (cycles). */
    void fetchMiss(Cycles start, Cycles end, InstSeqNum next_seq);

    /** A correct-path instruction committed. */
    void commit(const InstRec &r);

    /**
     * Queue the trace line of the instruction just committed (everything
     * before "START").  The previous queued line is written with a
     * placeholder stack; the last one is written by finish().
     */
    void emitLine(std::string prefix);

    /** Classify all slots and write the final trace line. */
    void finish(std::vector<InstRec> live);

    bool active() const { return started; }

  private:
    static int rowOf(int stage);
    Result compute(std::vector<InstRec> &live);
    void writeStack(const std::string &prefix, const Result &r);

    int W = 1;
    int delayCfg[4] = {-1, -1, -1, -1};
    FILE *tracePtr = nullptr;
    std::string diagPath;

    bool started = false;
    bool finished = false;
    int64_t c0 = 0;

    std::vector<InstRec> committed;
    std::vector<uint16_t> decProc;     // per relative cycle
    std::vector<uint8_t> ticked;       // per relative cycle: CPU ticked
    // per row, per cycle
    std::vector<InstSeqNum> heldSeq[NumRows];  // 0 = none
    std::vector<uint8_t> heldSub[NumRows];
    std::vector<uint8_t> stallSub[NumRows];    // BPSub + 1; 0 = none

    struct Miss { int64_t start, end; InstSeqNum nextSeq; };
    std::vector<Miss> misses;

    std::string pendingLine;
    bool havePending = false;
};

} // namespace o3
} // namespace gem5

#endif // __CPU_O3_MSCS_HH__
