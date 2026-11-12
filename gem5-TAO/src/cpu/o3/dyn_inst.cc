/*
 * Copyright (c) 2010-2011, 2021 ARM Limited
 * All rights reserved
 *
 * The license below extends only to copyright in the software and shall
 * not be construed as granting a license to any other intellectual
 * property including but not limited to intellectual property relating
 * to a hardware implementation of the functionality of the software
 * licensed hereunder.  You may use the software subject to the license
 * terms below provided that you ensure that this notice is replicated
 * unmodified and in its entirety in all distributions of the software,
 * modified or unmodified, in source code or in binary form.
 *
 * Copyright (c) 2004-2005 The Regents of The University of Michigan
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met: redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer;
 * redistributions in binary form must reproduce the above copyright
 * notice, this list of conditions and the following disclaimer in the
 * documentation and/or other materials provided with the distribution;
 * neither the name of the copyright holders nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "cpu/o3/dyn_inst.hh"

#include <algorithm>

#include "base/intmath.hh"
#include "debug/DynInst.hh"
#include "debug/IQ.hh"
#include "debug/O3PipeView.hh"

#include "cpu/o3/fetch.hh"
#include "cpu/o3/decode.hh"
#include "cpu/o3/rename.hh"
#include "cpu/o3/iew.hh"
#include "cpu/base.hh"
#include "cpu/o3/lsq.hh"
#include "cpu/o3/inst_queue.hh"

#include "cpu/pred/bpred_unit.hh"

namespace gem5
{

namespace o3
{

DynInst::DynInst(const Arrays &arrays, const StaticInstPtr &static_inst,
        const StaticInstPtr &_macroop, InstSeqNum seq_num, CPU *_cpu)
    : seqNum(seq_num), staticInst(static_inst), cpu(_cpu),
      _numSrcs(arrays.numSrcs), _numDests(arrays.numDests),
      _flatDestIdx(arrays.flatDestIdx), _destIdx(arrays.destIdx),
      _prevDestIdx(arrays.prevDestIdx), _srcIdx(arrays.srcIdx),
      _readySrcIdx(arrays.readySrcIdx), macroop(_macroop)
{
    std::fill(_readySrcIdx, _readySrcIdx + (numSrcs() + 7) / 8, 0);

    status.reset();

    instFlags.reset();
    instFlags[RecordResult] = true;
    instFlags[Predicate] = true;
    instFlags[MemAccPredicate] = true;

#ifndef NDEBUG
    ++cpu->instcount;

    if (cpu->instcount > 1500) {
#ifdef GEM5_DEBUG
        cpu->dumpInsts();
        dumpSNList();
#endif
        assert(cpu->instcount <= 1500);
    }

    DPRINTF(DynInst,
        "DynInst: [sn:%lli] Instruction created. Instcount for %s = %i\n",
        seqNum, cpu->name(), cpu->instcount);
#endif

#ifdef GEM5_DEBUG
    cpu->snList.insert(seqNum);
#endif

}

DynInst::DynInst(const Arrays &arrays, const StaticInstPtr &static_inst,
        const StaticInstPtr &_macroop, const PCStateBase &_pc,
        const PCStateBase &pred_pc, InstSeqNum seq_num, CPU *_cpu)
    : DynInst(arrays, static_inst, _macroop, seq_num, _cpu)
{
    set(pc, _pc);
    set(predPC, pred_pc);
}

DynInst::DynInst(const Arrays &arrays, const StaticInstPtr &_staticInst,
        const StaticInstPtr &_macroop)
    : DynInst(arrays, _staticInst, _macroop, 0, nullptr)
{}

/*
 * This custom "new" operator uses the default "new" operator to allocate space
 * for a DynInst, but also pads out the number of bytes to make room for some
 * extra structures the DynInst needs. We save time and improve performance by
 * only going to the heap once to get space for all these structures.
 *
 * When a DynInst is allocated with new, the compiler will call this "new"
 * operator with "count" set to the number of bytes it needs to store the
 * DynInst. We ultimately call into the default new operator to get those
 * bytes, but before we do, we pad out "count" so that there will be extra
 * space for some structures the DynInst needs. We take into account both the
 * absolute size of these structures, and also what alignment they need.
 *
 * Once we've gotten a buffer large enough to hold the DynInst itself and these
 * extra structures, we construct the extra bits using placement new. This
 * constructs the structures in place in the space we created for them.
 *
 * Next, we return the buffer as the result of our operator. The compiler takes
 * that buffer and constructs the DynInst in the beginning of it using the
 * DynInst constructor.
 *
 * To avoid having to calculate where these extra structures are twice, once
 * when making room for them and initializing them, and then once again in the
 * DynInst constructor, we also pass in a structure called "arrays" which holds
 * pointers to them. The fields of "arrays" are initialized in this operator,
 * and are then consumed in the DynInst constructor.
 */
void *
DynInst::operator new(size_t count, Arrays &arrays)
{
    // Convenience variables for brevity.
    const auto num_dests = arrays.numDests;
    const auto num_srcs = arrays.numSrcs;

    // Figure out where everything will go.
    uintptr_t inst = 0;
    size_t inst_size = count;

    uintptr_t flat_dest_idx = roundUp(inst + inst_size, alignof(RegId));
    size_t flat_dest_idx_size = sizeof(*arrays.flatDestIdx) * num_dests;

    uintptr_t dest_idx =
        roundUp(flat_dest_idx + flat_dest_idx_size, alignof(PhysRegIdPtr));
    size_t dest_idx_size = sizeof(*arrays.destIdx) * num_dests;

    uintptr_t prev_dest_idx =
        roundUp(dest_idx + dest_idx_size, alignof(PhysRegIdPtr));
    size_t prev_dest_idx_size = sizeof(*arrays.prevDestIdx) * num_dests;

    uintptr_t src_idx =
        roundUp(prev_dest_idx + prev_dest_idx_size, alignof(PhysRegIdPtr));
    size_t src_idx_size = sizeof(*arrays.srcIdx) * num_srcs;

    uintptr_t ready_src_idx =
        roundUp(src_idx + src_idx_size, alignof(uint8_t));
    size_t ready_src_idx_size =
        sizeof(*arrays.readySrcIdx) * ((num_srcs + 7) / 8);

    // Figure out how much space we need in total.
    size_t total_size = ready_src_idx + ready_src_idx_size;

    // Actually allocate it.
    uint8_t *buf = (uint8_t *)::operator new(total_size);

    // Fill in "arrays" with pointers to all the arrays.
    arrays.flatDestIdx = (RegId *)(buf + flat_dest_idx);
    arrays.destIdx = (PhysRegIdPtr *)(buf + dest_idx);
    arrays.prevDestIdx = (PhysRegIdPtr *)(buf + prev_dest_idx);
    arrays.srcIdx = (PhysRegIdPtr *)(buf + src_idx);
    arrays.readySrcIdx = (uint8_t *)(buf + ready_src_idx);

    // Initialize all the extra components.
    new (arrays.flatDestIdx) RegId[num_dests];
    new (arrays.destIdx) PhysRegIdPtr[num_dests];
    new (arrays.prevDestIdx) PhysRegIdPtr[num_dests];
    new (arrays.srcIdx) PhysRegIdPtr[num_srcs];
    new (arrays.readySrcIdx) uint8_t[num_srcs];

    return buf;
}

// Because of the custom "new" operator that allocates more bytes than the
// size of the DynInst object, AddressSanitizer throw new-delete-type-mismatch.
// Adding a custom delete function is enough to shut down this false positive
void
DynInst::operator delete(void *ptr)
{
    ::operator delete(ptr);
}

DynInst::~DynInst()
{
    /*
     * The buffer this DynInst occupies also holds some of the structures it
     * points to. We need to call their destructors manually to make sure that
     * they're cleaned up appropriately, but we don't need to free their memory
     * explicitly since that's part of the DynInst's buffer and is already
     * going to be freed as part of deleting the DynInst.
     */
    for (int i = 0; i < _numDests; i++) {
        _flatDestIdx[i].~RegId();
        _destIdx[i].~PhysRegIdPtr();
        _prevDestIdx[i].~PhysRegIdPtr();
    }

    for (int i = 0; i < _numSrcs; i++)
        _srcIdx[i].~PhysRegIdPtr();

    for (int i = 0; i < ((_numSrcs + 7) / 8); i++)
        _readySrcIdx[i].~uint8_t();

#if TRACING_ON
    if (debug::O3PipeView) {
        Tick fetch = fetchTick;
        // fetchTick can be -1 if the instruction fetched outside the trace
        // window.
        if (fetch != -1) {
            Tick val;
            // Print info needed by the pipeline activity viewer.
            DPRINTFR(O3PipeView, "O3PipeView:fetch:%llu:0x%08llx:%d:%llu:%s\n",
                     fetch,
                     pcState().instAddr(),
                     pcState().microPC(),
                     seqNum,
                     staticInst->disassemble(pcState().instAddr()));

            val = (decodeTick == -1) ? 0 : fetch + decodeTick;
            DPRINTFR(O3PipeView, "O3PipeView:decode:%llu\n", val);
            val = (renameTick == -1) ? 0 : fetch + renameTick;
            DPRINTFR(O3PipeView, "O3PipeView:rename:%llu\n", val);
            val = (dispatchTick == -1) ? 0 : fetch + dispatchTick;
            DPRINTFR(O3PipeView, "O3PipeView:dispatch:%llu\n", val);
            val = (issueTick == -1) ? 0 : fetch + issueTick;
            DPRINTFR(O3PipeView, "O3PipeView:issue:%llu\n", val);
            val = (completeTick == -1) ? 0 : fetch + completeTick;
            DPRINTFR(O3PipeView, "O3PipeView:complete:%llu\n", val);
            val = (commitTick == -1) ? 0 : fetch + commitTick;

            Tick valS = (storeTick == -1) ? 0 : fetch + storeTick;
            DPRINTFR(O3PipeView, "O3PipeView:retire:%llu:store:%llu\n",
                    val, valS);
        }
    }
#endif

    delete [] memData;
    delete traceData;
    fault = NoFault;

#ifndef NDEBUG
    --cpu->instcount;

    DPRINTF(DynInst,
        "DynInst: [sn:%lli] Instruction destroyed. Instcount for %s = %i\n",
        seqNum, cpu->name(), cpu->instcount);
#endif
#ifdef GEM5_DEBUG
    cpu->snList.erase(seqNum);
#endif
};


#ifdef GEM5_DEBUG
void
DynInst::dumpSNList()
{
    std::set<InstSeqNum>::iterator sn_it = cpu->snList.begin();

    int count = 0;
    while (sn_it != cpu->snList.end()) {
        cprintf("%i: [sn:%lli] not destroyed\n", count, (*sn_it));
        count++;
        sn_it++;
    }
}
#endif

void
DynInst::dump()
{
    cprintf("T%d : %#08d `", threadNumber, pc->instAddr());
    std::cout << staticInst->disassemble(pc->instAddr());
    cprintf("'\n");
}

void
DynInst::dump(std::string &outstring)
{
    std::ostringstream s;
    s << "T" << threadNumber << " : 0x" << pc->instAddr() << " "
      << staticInst->disassemble(pc->instAddr());

    outstring = s.str();
}

void
DynInst::markSrcRegReady()
{
    DPRINTF(IQ, "[sn:%lli] has %d ready out of %d sources. RTI %d)\n",
            seqNum, readyRegs+1, numSrcRegs(), readyToIssue());
    if (++readyRegs == numSrcRegs()) {
        setCanIssue();
    }
}

void
DynInst::markSrcRegReady(RegIndex src_idx)
{
    readySrcIdx(src_idx, true);
    markSrcRegReady();
}


void
DynInst::setSquashed()
{
    status.set(Squashed);

    if (!isPinnedRegsRenamed() || isPinnedRegsSquashDone())
        return;

    // This inst has been renamed already so it may go through rename
    // again (e.g. if the squash is due to memory access order violation).
    // Reset the write counters for all pinned destination register to ensure
    // that they are in a consistent state for a possible re-rename. This also
    // ensures that dest regs will be pinned to the same phys register if
    // re-rename happens.
    for (int idx = 0; idx < numDestRegs(); idx++) {
        PhysRegIdPtr phys_dest_reg = renamedDestIdx(idx);
        if (phys_dest_reg->isPinned()) {
            phys_dest_reg->incrNumPinnedWrites();
            if (isPinnedRegsWritten())
                phys_dest_reg->incrNumPinnedWritesToComplete();
        }
    }
    setPinnedRegsSquashDone();
}

Fault
DynInst::execute()
{
    // @todo: Pretty convoluted way to avoid squashing from happening
    // when using the TC during an instruction's execution
    // (specifically for instructions that have side-effects that use
    // the TC).  Fix this.
    bool no_squash_from_TC = thread->noSquashFromTC;
    thread->noSquashFromTC = true;

    fault = staticInst->execute(this, traceData);

    thread->noSquashFromTC = no_squash_from_TC;

    return fault;
}

Fault
DynInst::initiateAcc()
{
    // @todo: Pretty convoluted way to avoid squashing from happening
    // when using the TC during an instruction's execution
    // (specifically for instructions that have side-effects that use
    // the TC).  Fix this.
    bool no_squash_from_TC = thread->noSquashFromTC;
    thread->noSquashFromTC = true;

    fault = staticInst->initiateAcc(this, traceData);

    thread->noSquashFromTC = no_squash_from_TC;

    return fault;
}

Fault
DynInst::completeAcc(PacketPtr pkt)
{
    // @todo: Pretty convoluted way to avoid squashing from happening
    // when using the TC during an instruction's execution
    // (specifically for instructions that have side-effects that use
    // the TC).  Fix this.
    bool no_squash_from_TC = thread->noSquashFromTC;
    thread->noSquashFromTC = true;

    if (cpu->checker) {
        if (isStoreConditional()) {
            reqToVerify->setExtraData(pkt->req->getExtraData());
        }
    }

    fault = staticInst->completeAcc(pkt, this, traceData);

    thread->noSquashFromTC = no_squash_from_TC;

    return fault;
}

// 1,"  movz   x29, #0, #0",2916,0,movz,0,0,0,0,0,0,,-5+6 done-0,0,0,0,,-4 done-0,0,0,0,0,1,1,,-7 done-0,0,0,0,0,0,0,0,,-numsrc done 829,,-num dest done 6-78,2,2,0,0,-Fetch Start-,19,0,0,0,0,0,0,1,0,-Fetch Done-,18,0,0,1,,-Decode Done-18,0,0,1,0,0,,-Rename Done-0,0,0,1,0,0,,-IEW Done-0,0,0,0,,-LSQ Done-18,0,1,0,0,1,0,0,0,,-Issue Done-0,0,0,0,0,0,0.000000,,-IQ stat Done-0,1,0,0,0,0

// void
// DynInst::dumpInst(FILE *trace_ptr) {
// 	fprintf(trace_ptr, "%ld,\"%s\",%ld,%ld,%s,%d,%u,%d,%d,%d,%d,",
//         this->seqNum, staticInst->disassemble(this->pcState().instAddr()).c_str() ,this->pcState().instAddr(), this->effAddrValid()?this->effAddr:0,this->staticInst->getName().c_str(),//5
// 	this->isMemRef(), this->effAddrValid() ? this->effSize : 0,this->isNop(), this->isNonSpeculative(), this->isControl(), this->isDirectCtrl()); // 6
    
//     // 106,"  ldr   x3, [sp, #104]",12656708,0,ldr
//     // 1,0,0,0,0,0
//     // fprintf(trace_ptr, ",-5+6 done-");
	
// 	fprintf(trace_ptr,"%d,%d,%d,%d,",
// 		this->isIndirectCtrl(), this->isCondCtrl(), this->isUncondCtrl(), cachedepth);   // 4
//     // 0,0,0,0,

//         // fprintf(trace_ptr, ",-4 done-");
        
// 	 fprintf(trace_ptr,"%d,%d,%d,%d,%lu,%lu,%d,",
// 		this->mispredicted(),this->isSquashed(), this->isMacroop(), this->isMicroop(), this->numSrcs(), this->numDests(), this->opClass());  // 7
//     // 0,0,0,0,1,1,52,
//         // fprintf(trace_ptr, ",-7 done-");

// 	for (int i = 0; i < this->numSrcs(); i++){
// 	    fprintf(trace_ptr,"%d,", this->staticInst->srcRegIdx(i).index());
// 	}

// 	int remain = 8- this->numSrcs(); 
// 	for (int i = 0; i <remain; i++){fprintf(trace_ptr,"0,");}
    
//     // fprintf(trace_ptr, ",-numsrc done 8");

//  	for (int i = 0; i < this->numDests(); i++){
// 	     fprintf(trace_ptr, "%d,", this->staticInst->destRegIdx(i).index());}
// 	remain = 6- this->numDests(); // 8 + 6 => 22 + 14 = 36
    
//     for (int i = 0; i <remain; i++){fprintf(trace_ptr,"0,");}

//     // fprintf(trace_ptr, ",-num dest done 6-");

	
//     // for (int i = 0; i < remain; i++){fprintf(trace_ptr,"0,");}
// 	// if (fetchTick){fetchTick=fetchTick/1000;}
// 	// if (issueTick){issueTick=issueTick/1000;}
// 	// if (completeTick){completeTick=completeTick/1000;}
// 	// if (commitTick){commitTick=commitTick/1000;}
// 	// if (storeTick){storeTick=storeTick/1000;}
// 	// fprintf(trace_ptr, "%lu,%d,%d,%d,%d",fetchTick,issueTick, completeTick, commitTick, storeTick);	

//     uint64_t fetch_us = fetchTick ? fetchTick / 1000 : 0;
//     uint64_t issue_us = issueTick ? issueTick / 1000 : 0;
//     uint64_t complete_us = completeTick ? completeTick / 1000 : 0;
//     uint64_t commit_us = commitTick ? commitTick / 1000 : 0;
//     uint64_t store_us = storeTick ? storeTick / 1000 : 0;
    
//     fprintf(trace_ptr, "%lu,%lu,%lu,%lu,%lu", 
//             fetch_us, issue_us, complete_us, commit_us, store_us); // 36 + 5 = 41
    
//     // fprintf(trace_ptr, ",-Fetch Start-");
//     // fetch stat data
//     // auto& dist = fetchStats.nisnDist;
//     Fetch &fetch_stat = this->cpu->fetch;
//     for (size_t i = 0; i < fetch_stat.localNisnDist.size(); ++i) {
//         uint64_t val = fetch_stat.localNisnDist[i];
//         fprintf(trace_ptr, ",%ld", val);
//         // if (i == fetc_stat.fetchWidth) {fprintf(trace_ptr, "\n");}
//     } // 41 + 9 = 50
//     // fetch_stat.initNisnVector();
//     // fprintf(trace_ptr, ",-Fetch Done-");
//     // decode stat data
//     Decode &decode_stat = this->cpu->decode;
//     fprintf(trace_ptr,",%ld,%ld,%ld,%ld,", 
//         decode_stat.idleCycles_stat, 
//         decode_stat.blockedCycles_stat, 
//         decode_stat.unblockCycles_stat, 
//         decode_stat.squashCycles_stat); // 50 + 4 = 54
//     // decode_stat.init_decode_stat();
    
//     // fprintf(trace_ptr, ",-Decode Done-");
//     // rename stat data
//     Rename &rename_stat = this->cpu->rename;
//     fprintf(trace_ptr,"%ld,%ld,%ld,%ld,%ld,%ld,", 
//         rename_stat.idleCycles_stat, 
//         rename_stat.blockCycles_stat, 
//         rename_stat.unblockCycles_stat, 
//         rename_stat.squashCycles_stat, 
//         rename_stat.IQFullEvents_stat, 
//         rename_stat.SQFullEvents_stat); // 54 + 6 = 60
//     // rename_stat.init_rename_stat();
    
//     // fprintf(trace_ptr, ",-Rename Done-");
//     // iew stat data
//     IEW &iew_stat = this->cpu->iew;
//     fprintf(trace_ptr,"%ld,%ld,%ld,%ld,%ld,%ld,", 
//         iew_stat.idleCycles_stat, 
//         iew_stat.blockCycles_stat, 
//         iew_stat.unblockCycles_stat, 
//         iew_stat.squashCycles_stat, 
//         iew_stat.iqFullEvents_stat, 
//         iew_stat.lsqFullEvents_stat); // 60 + 6 = 66
//     // iew_stat.init_iew_stat();
    
//     // fprintf(trace_ptr, ",-IEW Done-");
//     // lsq stat data
//     LSQ& lsq_stat = this->cpu->iew.ldstQueue;
//     lsq_stat.dumpLSQUnitStats(trace_ptr, 0); // 66 + 4 = 70
    
//     // fprintf(trace_ptr, ",-LSQ Done-");
//     // iq stat data issue_NisnDist
//     InstructionQueue& iq_stat = this->cpu->iew.instQueue;
//     for (size_t i = 0; i < iq_stat.issue_NisnDist.size(); ++i) {
//         uint64_t val = iq_stat.issue_NisnDist[i];
//         fprintf(trace_ptr, "%ld,", val);
//         // if (i == iq_stat.issue_NisnDist.size()-1) {fprintf(trace_ptr, "\n");}
//     } // 70 + 9 = 79
//     // iq_stat.initNisnVector();
    
//     // fprintf(trace_ptr, ",-Issue Done-");
//     // fubusy stat data
//     // fprintf(trace_ptr, "%ld,%ld,%ld,%ld,%ld,%ld,%.6f,", 
//     //     iq_stat.statFuBusy_intalu,
//     //     iq_stat.statFuBusy_intmult,
//     //     iq_stat.statFuBusy_floatadd,
//     //     iq_stat.statFuBusy_floatmult,
//     //     iq_stat.statFuBusy_memread,
//     //     iq_stat.statFuBusy_memwrite,
//     //     iq_stat.getFUBusyRate()); // 79 + 7 = 86
    
//         // fprintf(trace_ptr, ",-IQ stat Done-");
//     // branch predictor stat data
//     gem5::branch_prediction::BPredUnit* bp_stat = fetch_stat.branchPred;
//     fprintf(trace_ptr, "%lu,%lu,%lu,%lu,%lu,%lu\n",
//         bp_stat->condPredicted_stat,
//         bp_stat->condIncorrect_stat, 
//         bp_stat->getBTBHitRatio(),
//         bp_stat->RASUsed_stat,
//         bp_stat->RASIncorrect_stat,
//         bp_stat->indirectMispredicted_stat);
    
//         // fprintf(trace_ptr, ",-BP Done-");
//     // dcache stat data

// }

bool
DynInst::isBranchTaken()
{
    // 1. 不是控制指令，直接返回 false
    if (!this->isControl()) {
        return false;
    }

    // 2. 无条件控制流指令 (Call, Jump, Ret 等)：总是 Taken
    // 注意：gem5 中 UncondCtrl 通常包含直接和间接跳转，它们在定义上都是 Taken 的
    if (this->isUncondCtrl()) {
        return true;
    }

    // 3. 条件分支：利用预测状态和误预测位来推断实际结果
    // 只有在 Commit 阶段 dump 时，mispredicted() 才是最终确定的
    // 逻辑：(预测跳转 XOR 误预测) == 实际跳转
    return this->readPredTaken() ^ this->mispredicted();
}

// void
// DynInst::dumpInst_fourstate(FILE *trace_ptr) {
// 	fprintf(trace_ptr, "%ld,\"%s\",%ld,%ld,%s,%d,%u,%d,%d,%d,%d,",
//         this->seqNum, staticInst->disassemble(this->pcState().instAddr()).c_str() ,this->pcState().instAddr(), this->effAddrValid()?this->effAddr:0,this->staticInst->getName().c_str(),//5
// 	this->isMemRef(), this->effAddrValid() ? this->effSize : 0,this->isNop(), this->isNonSpeculative(), this->isControl(), this->isDirectCtrl()); // 6
    
//     // 106, "  ldr   x3, [sp, #104]", 12656708,0,ldr
//     // 1,0,0,0,0,0 
//     // total 11
	
// 	fprintf(trace_ptr,"%d,%d,%d,%d,%d,",
// 		this->isIndirectCtrl(), this->isCondCtrl(), this->isUncondCtrl(), cachedepth, isBranchTaken());   // 4
//     // 0,0,0,0,
//     // total 4
        
// 	 fprintf(trace_ptr,"%d,%d,%d,%d,%lu,%lu,%d,",
// 		this->mispredicted(),this->isSquashed(), this->isMacroop(), this->isMicroop(), this->numSrcs(), this->numDests(), this->opClass());  // 7
//     // 0,0,0,0,1,1,52,

// 	for (int i = 0; i < this->numSrcs(); i++){
// 	    fprintf(trace_ptr,"%d,", this->staticInst->srcRegIdx(i).index());
// 	}

// 	int remain = 8- this->numSrcs(); 
// 	for (int i = 0; i <remain; i++){fprintf(trace_ptr,"0,");}

//  	for (int i = 0; i < this->numDests(); i++){
// 	     fprintf(trace_ptr, "%d,", this->staticInst->destRegIdx(i).index());}
// 	remain = 6- this->numDests(); // 8 + 6 => 22 + 14 = 36
    
//     for (int i = 0; i <remain; i++){fprintf(trace_ptr,"0,");}

//     // for (int i = 0; i < remain; i++){fprintf(trace_ptr,"0,");}
// 	// if (fetchTick){fetchTick=fetchTick/1000;}
// 	// if (issueTick){issueTick=issueTick/1000;}
// 	// if (completeTick){completeTick=completeTick/1000;}
// 	// if (commitTick){commitTick=commitTick/1000;}
// 	// if (storeTick){storeTick=storeTick/1000;}
// 	// fprintf(trace_ptr, "%lu,%d,%d,%d,%d",fetchTick,issueTick, completeTick, commitTick, storeTick);	

//     uint64_t fetch_us = fetchTick ? fetchTick / 1000 : 0;
//     uint64_t issue_us = issueTick ? issueTick / 1000 : 0;
//     uint64_t complete_us = completeTick ? completeTick / 1000 : 0;
//     uint64_t commit_us = commitTick ? commitTick / 1000 : 0;
//     uint64_t store_us = storeTick ? storeTick / 1000 : 0;
    
//     fprintf(trace_ptr, "%lu,%lu,%lu,%lu,%lu", 
//             fetch_us, issue_us, complete_us, commit_us, store_us); // 36 + 5 = 41
    
//     // fprintf(trace_ptr, ",-Fetch Start-");
//     // fetch stat data
//     // auto& dist = fetchStats.nisnDist;
//     // Fetch &fetch_stat = this->cpu->fetch;
//     // for (size_t i = 0; i < fetch_stat.localNisnDist.size(); ++i) {
//     //     uint64_t val = fetch_stat.localNisnDist[i];
//     //     fprintf(trace_ptr, ",%ld", val);
//     // }
//         // if (i == fetc_stat.fetchWidth) {fprintf(trace_ptr, "\n");}
//      // 41 + 9 = 50
//     // fetch_stat.initNisnVector();
//     // fprintf(trace_ptr, ",-Fetch Done-");
//     // decode stat data
    
//     Decode &decode_stat = this->cpu->decode;
//     fprintf(trace_ptr,"START,%ld,%ld,%ld,%ld,", 
//         decode_stat.runCycles_stat,
//         decode_stat.idleCycles_stat, 
//         decode_stat.blockedCycles_stat, 
//         decode_stat.squashCycles_stat); // 41 + 4 = 45
    
//     // rename stat data
//     Rename &rename_stat = this->cpu->rename;
//     fprintf(trace_ptr,"%ld,%ld,%ld,%ld,%ld,%ld,", 
//         rename_stat.runCycles_stat,
//         rename_stat.idleCycles_stat, 
//         rename_stat.blockCycles_stat,
//         rename_stat.otherstall_reg_stat,
//         rename_stat.otherstall_serialize_stat,
//         rename_stat.squashCycles_stat); // 45 + 6 = 51
    
//     // iew stat data
//     IEW &iew_stat = this->cpu->iew;
//     fprintf(trace_ptr,"%ld,%ld,%ld,%ld,%ld,%ld,", 
//         iew_stat.runCycles_stat, // 10
//         iew_stat.idleCycles_stat,  // 11
//         iew_stat.blockCycles_stat, // 12
//         iew_stat.blockCycles_iq_stat,  // 13
//         iew_stat.blockCycles_lsq_stat,  // 14
//         iew_stat.squashCycles_stat); // 15 // 51 + 6 = 57
    
//     // iq stat data
//     InstructionQueue &iq_stat = iew_stat.instQueue;
//     fprintf(trace_ptr,"%ld,%ld,%ld,%ld,%ld,%ld,", 
//         iq_stat.runCycles_stat, // 16
//         iq_stat.idleCycles_stat,  // 17
//         iq_stat.blockCycles_depend_stat,  // 18
//         iq_stat.blockCycles_mem_stat,  // 19
//         iq_stat.blockCycles_fu_stat, // 20
//         iq_stat.squashCycles_stat); // 21 // 51 + 6 = 57
    
//     // commit stat data
//     Commit &commit_stat = this->cpu->commit;
//     fprintf(trace_ptr,"%ld,%ld,%ld,%ld,%ld,%ld\n", 
//         commit_stat.runCycles_stat,
//         commit_stat.idleCycles_stat, 
//         commit_stat.blockCycles_fu_stat, 
//         commit_stat.blockCycles_mem_stat, 
//         commit_stat.blockCycles_depend_stat, 
//         commit_stat.squashCycles_stat); // 57 + 6 = 63
        
//     // fprintf(trace_ptr, ",-IEW Done-");
//     // lsq stat data
//     // LSQ& lsq_stat = this->cpu->iew.ldstQueue;
//     // lsq_stat.dumpLSQUnitStats(trace_ptr, 0); // 66 + 4 = 70
    
//     // fprintf(trace_ptr, ",-LSQ Done-");
//     // iq stat data issue_NisnDist
//     // InstructionQueue& iq_stat = this->cpu->iew.instQueue;
//     // for (size_t i = 0; i < iq_stat.issue_NisnDist.size(); ++i) {
//     //     uint64_t val = iq_stat.issue_NisnDist[i];
//     //     fprintf(trace_ptr, "%ld,", val);
//         // if (i == iq_stat.issue_NisnDist.size()-1) {fprintf(trace_ptr, "\n");}
//     // } // 70 + 9 = 79
//     // iq_stat.initNisnVector();
    
//     // fprintf(trace_ptr, ",-Issue Done-");
//     // fubusy stat data
//     // fprintf(trace_ptr, "%ld,%ld,%ld,%ld,%ld,%ld,%.6f,", 
//     //     iq_stat.statFuBusy_intalu,
//     //     iq_stat.statFuBusy_intmult,
//     //     iq_stat.statFuBusy_floatadd,
//     //     iq_stat.statFuBusy_floatmult,
//     //     iq_stat.statFuBusy_memread,
//     //     iq_stat.statFuBusy_memwrite,
//     //     iq_stat.getFUBusyRate()); // 79 + 7 = 86
    
//         // fprintf(trace_ptr, ",-IQ stat Done-");
//     // branch predictor stat data
//     // gem5::branch_prediction::BPredUnit* bp_stat = fetch_stat.branchPred;
//     // fprintf(trace_ptr, "%lu,%lu,%lu,%lu,%lu,%lu\n",
//     //     bp_stat->condPredicted_stat,
//     //     bp_stat->condIncorrect_stat, 
//     //     bp_stat->getBTBHitRatio(),
//     //     bp_stat->RASUsed_stat,
//     //     bp_stat->RASIncorrect_stat,
//     //     bp_stat->indirectMispredicted_stat);
    
//         // fprintf(trace_ptr, ",-BP Done-");
//     // dcache stat data
// }

void
DynInst::dumpInst_fourstate(FILE *trace_ptr) {
	fprintf(trace_ptr, "%ld,\"%s\",%ld,%ld,%s,%d,%u,%d,%d,%d,%d,",
        this->seqNum, staticInst->disassemble(this->pcState().instAddr()).c_str() ,this->pcState().instAddr(), this->effAddrValid()?this->effAddr:0,this->staticInst->getName().c_str(),//5
	this->isMemRef(), this->effAddrValid() ? this->effSize : 0,this->isNop(), this->isNonSpeculative(), this->isControl(), this->isDirectCtrl()); // 6

	fprintf(trace_ptr,"%d,%d,%d,%d,%d,",
		this->isIndirectCtrl(), this->isCondCtrl(), this->isUncondCtrl(), cachedepth, isBranchTaken());   // 4
        
	 fprintf(trace_ptr,"%d,%d,%d,%d,%lu,%lu,%d,",
		this->mispredicted(),this->isSquashed(), this->isMacroop(), this->isMicroop(), this->numSrcs(), this->numDests(), this->opClass());  // 7

	for (int i = 0; i < this->numSrcs(); i++){
	    fprintf(trace_ptr,"%d,", this->staticInst->srcRegIdx(i).index());
	}

	int remain = 8- this->numSrcs(); 
	for (int i = 0; i <remain; i++){fprintf(trace_ptr,"0,");}

 	for (int i = 0; i < this->numDests(); i++){
	     fprintf(trace_ptr, "%d,", this->staticInst->destRegIdx(i).index());}
	remain = 6- this->numDests(); // 8 + 6 => 22 + 14 = 36
    
    for (int i = 0; i <remain; i++){fprintf(trace_ptr,"0,");}

    uint64_t fetch_us = fetchTick ? fetchTick / 1000 : 0;
    uint64_t issue_us = issueTick ? issueTick / 1000 : 0;
    uint64_t complete_us = completeTick ? completeTick / 1000 : 0;
    uint64_t commit_us = commitTick ? commitTick / 1000 : 0;
    uint64_t store_us = storeTick ? storeTick / 1000 : 0;
    
    fprintf(trace_ptr, "%lu,%lu,%lu,%lu,%lu", 
            fetch_us, issue_us, complete_us, commit_us, store_us); // 36 + 5 = 41

    // The MSCS fields ("START,...") are appended by MSCSAccountant
    // (cpu/o3/mscs.cc). The per-stage counter accounting below is the
    // previous implementation, kept for reference only.
#if 0
    Decode &decode_stat = this->cpu->decode;
    Rename &rename_stat = this->cpu->rename;
    IEW &iew_stat = this->cpu->iew;
    InstructionQueue &iq_stat = iew_stat.instQueue;
    Commit &commit_stat = this->cpu->commit;

    // int W_min = this->cpu->W_min;
    // long sleep_fe = this->cpu->sleepFrontendSlots * W_min;
    // long sleep_be = this->cpu->sleepBackendSlots * W_min;
    // long commit_base = commit_stat.runCycles_stat;

    // // Decode
    // long decode_badspec = decode_stat.runCycles_stat - commit_base;
    // fprintf(trace_ptr,"START,%ld,%ld,%ld,%ld,", 
    //     commit_base,
    //     decode_stat.idleCycles_stat + sleep_fe, 
    //     decode_stat.blockedCycles_stat + sleep_be, 
    //     decode_stat.squashCycles_stat + decode_badspec);
    
    // // Rename (dispatch)
    // long rename_badspec = rename_stat.runCycles_stat - commit_base;
    // fprintf(trace_ptr,"%ld,%ld,%ld,%ld,%ld,%ld,", 
    //     commit_base,
    //     rename_stat.idleCycles_stat + sleep_fe, 
    //     rename_stat.blockCycles_stat + sleep_be,
    //     rename_stat.otherstall_reg_stat,
    //     rename_stat.otherstall_serialize_stat,
    //     rename_stat.squashCycles_stat + rename_badspec);
    
    // // IEW dispatch
    // long iew_badspec = iew_stat.runCycles_stat - commit_base;
    // fprintf(trace_ptr,"%ld,%ld,%ld,%ld,%ld,%ld,", 
    //     commit_base,
    //     iew_stat.idleCycles_stat + sleep_fe,
    //     iew_stat.blockCycles_stat + sleep_be,
    //     iew_stat.blockCycles_iq_stat,
    //     iew_stat.blockCycles_lsq_stat,
    //     iew_stat.squashCycles_stat + iew_badspec);
    
    // // IQ (issue)
    // long iq_badspec = iq_stat.runCycles_stat - commit_base;
    // fprintf(trace_ptr,"%ld,%ld,%ld,%ld,%ld,%ld,", 
    //     commit_base,
    //     iq_stat.idleCycles_stat + sleep_fe,
    //     iq_stat.blockCycles_depend_stat + sleep_be,
    //     iq_stat.blockCycles_mem_stat,
    //     iq_stat.blockCycles_fu_stat,
    //     iq_stat.squashCycles_stat + iq_badspec);
    
    // // Commit
    // fprintf(trace_ptr,"%ld,%ld,%ld,%ld,%ld,%ld\n", 
    //     commit_base,
    //     commit_stat.idleCycles_stat + sleep_fe, 
    //     commit_stat.blockCycles_fu_stat + sleep_be, 
    //     commit_stat.blockCycles_mem_stat, 
    //     commit_stat.blockCycles_depend_stat, 
    //     commit_stat.squashCycles_stat);

////////////////////////////////////////////////////////////////////////////////////

    int W_min = this->cpu->W_min;
    // long sleep_fe = this->cpu->sleepFrontendSlots * W_min;
    // long sleep_be = this->cpu->sleepBackendSlots * W_min;
    long decode_sleep_fe   = this->cpu->sleepFE_decode   * W_min;
    long decode_sleep_be   = this->cpu->sleepBE_decode   * W_min;
    long dispatch_sleep_fe = this->cpu->sleepFE_dispatch * W_min;
    long dispatch_sleep_be = this->cpu->sleepBE_dispatch * W_min;
    long issue_sleep_fe    = this->cpu->sleepFE_issue    * W_min;
    long issue_sleep_be    = this->cpu->sleepBE_issue    * W_min;
    long commit_sleep_fe   = this->cpu->sleepFE_commit   * W_min;
    long commit_sleep_be   = this->cpu->sleepBE_commit   * W_min;

    // long commit_base = commit_stat.runCycles_stat;
    long commit_base = commit_stat.runCycles_stat - commit_stat.nonwork_base;
    long total_cycles = this->cpu->baseStats.numCycles.value();
    long expected_total = total_cycles * W_min;
    double E = (double)expected_total;

    // ---- Decode ----
    long decode_badspec = std::max((long)(decode_stat.runCycles_stat - commit_base), 0L);
    // long decode_badspec = std::max((long)(decode_stat.runCycles_stat - decode_stat.nonwork_base) - commit_base, 0L);

    // long decode_fe = decode_stat.idleCycles_stat + sleep_fe;
    // long decode_be = decode_stat.blockedCycles_stat + sleep_be;
    long decode_fe = decode_stat.idleCycles_stat + decode_sleep_fe;
    long decode_be = decode_stat.blockedCycles_stat + decode_sleep_be;
    long decode_sq = decode_stat.squashCycles_stat + decode_badspec;
    long decode_gap = expected_total - (commit_base + decode_fe + decode_be + decode_sq);
    decode_fe += decode_gap;

    fprintf(trace_ptr,"START,%.6f,%.6f,%.6f,%.6f,", 
        commit_base/E, decode_fe/E, decode_be/E, decode_sq/E);

    // ---- Rename (dispatch) ----
    long rename_badspec = std::max((long)(rename_stat.runCycles_stat - commit_base), 0L);

    // long rename_badspec = std::max((long)(rename_stat.runCycles_stat - rename_stat.nonwork_base) - commit_base, 0L);

    long rename_fe = rename_stat.idleCycles_stat + dispatch_sleep_fe;
    long rename_be_total = rename_stat.blockCycles_stat 
                         + rename_stat.otherstall_reg_stat 
                         + rename_stat.otherstall_serialize_stat + dispatch_sleep_be;
    long rename_sq = rename_stat.squashCycles_stat + rename_badspec;
    long rename_gap = expected_total - (commit_base + rename_fe + rename_be_total + rename_sq);
    rename_fe += rename_gap;

    fprintf(trace_ptr,"%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,", 
        commit_base/E, rename_fe/E, (rename_stat.blockCycles_stat + dispatch_sleep_be)/E,
        rename_stat.otherstall_reg_stat/E,
        rename_stat.otherstall_serialize_stat/E,
        rename_sq/E);

    // ---- IEW dispatch ----
    long iew_badspec = iew_stat.runCycles_stat - commit_base;
    long iew_fe = iew_stat.idleCycles_stat + issue_sleep_be;
    long iew_be_total = iew_stat.blockCycles_stat 
                      + iew_stat.blockCycles_iq_stat 
                      + iew_stat.blockCycles_lsq_stat + issue_sleep_be;
    long iew_sq = iew_stat.squashCycles_stat + iew_badspec;
    long iew_gap = expected_total - (commit_base + iew_fe + iew_be_total + iew_sq);
    iew_fe += iew_gap;

    fprintf(trace_ptr,"%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,", 
        commit_base/E, iew_fe/E, (iew_stat.blockCycles_stat + issue_sleep_be)/E,
        iew_stat.blockCycles_iq_stat/E,
        iew_stat.blockCycles_lsq_stat/E,
        iew_sq/E);

    // ---- IQ (issue) ----
    // long iq_badspec = iq_stat.runCycles_stat - commit_base;
    long iq_badspec_raw = std::max((long)(iq_stat.runCycles_stat - commit_base), 0L);
    // long iq_badspec_raw = std::max((long)(iq_stat.runCycles_stat - iq_stat.nonwork_base) - commit_base, 0L);
    long iq_badspec = std::min(iq_badspec_raw, rename_badspec);

    long iq_fe = iq_stat.idleCycles_stat + issue_sleep_fe;
    long iq_be_total = iq_stat.blockCycles_depend_stat 
                     + iq_stat.blockCycles_mem_stat 
                     + iq_stat.blockCycles_fu_stat + issue_sleep_be;
    long iq_sq = iq_stat.squashCycles_stat + iq_badspec;
    long iq_gap = expected_total - (commit_base + iq_fe + iq_be_total + iq_sq);
    iq_fe += iq_gap;

    fprintf(trace_ptr,"%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,", 
        commit_base/E, iq_fe/E, (iq_stat.blockCycles_depend_stat + issue_sleep_be)/E,
        iq_stat.blockCycles_mem_stat/E,
        iq_stat.blockCycles_fu_stat/E,
        iq_sq/E);

    // ---- Commit ----
    long commit_fe = commit_stat.idleCycles_stat + commit_sleep_fe;
    long commit_be_total = commit_stat.blockCycles_fu_stat 
                         + commit_stat.blockCycles_mem_stat 
                         + commit_stat.blockCycles_depend_stat +commit_sleep_be;
    long commit_sq = commit_stat.squashCycles_stat;
    long commit_gap = expected_total - (commit_base + commit_fe + commit_be_total + commit_sq);
    // commit_fe += commit_gap;
    commit_fe += commit_gap;

    fprintf(trace_ptr,"%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n", 
        commit_base/E, commit_fe/E, (commit_stat.blockCycles_fu_stat + commit_sleep_be)/E,
        commit_stat.blockCycles_mem_stat/E, 
        commit_stat.blockCycles_depend_stat/E, 
        commit_sq/E);
    
    ////////////////////////////////////////////////////////////////////////////////////
    
    // Decode &decode_stat = this->cpu->decode;
    // fprintf(trace_ptr,"START,%ld,%ld,%ld,%ld,", 
    //     decode_stat.runCycles_stat,
    //     decode_stat.idleCycles_stat, 
    //     decode_stat.blockedCycles_stat, 
    //     decode_stat.squashCycles_stat); // 41 + 4 = 45
    
    // // rename stat data
    // Rename &rename_stat = this->cpu->rename;
    // fprintf(trace_ptr,"%ld,%ld,%ld,%ld,%ld,%ld,", 
    //     rename_stat.runCycles_stat,
    //     rename_stat.idleCycles_stat, 
    //     rename_stat.blockCycles_stat,
    //     rename_stat.otherstall_reg_stat,
    //     rename_stat.otherstall_serialize_stat,
    //     rename_stat.squashCycles_stat); // 45 + 6 = 51
    
    // // iew stat data
    // IEW &iew_stat = this->cpu->iew;
    // fprintf(trace_ptr,"%ld,%ld,%ld,%ld,%ld,%ld,", 
    //     iew_stat.runCycles_stat, // 10
    //     iew_stat.idleCycles_stat,  // 11
    //     iew_stat.blockCycles_stat, // 12
    //     iew_stat.blockCycles_iq_stat,  // 13
    //     iew_stat.blockCycles_lsq_stat,  // 14
    //     iew_stat.squashCycles_stat); // 15 // 51 + 6 = 57
    
    // // iq stat data
    // InstructionQueue &iq_stat = iew_stat.instQueue;
    // fprintf(trace_ptr,"%ld,%ld,%ld,%ld,%ld,%ld,", 
    //     iq_stat.runCycles_stat, // 16
    //     iq_stat.idleCycles_stat,  // 17
    //     iq_stat.blockCycles_depend_stat,  // 18
    //     iq_stat.blockCycles_mem_stat,  // 19
    //     iq_stat.blockCycles_fu_stat, // 20
    //     iq_stat.squashCycles_stat); // 21 // 51 + 6 = 57
    
    // // commit stat data
    // Commit &commit_stat = this->cpu->commit;
    // fprintf(trace_ptr,"%ld,%ld,%ld,%ld,%ld,%ld\n", 
    //     commit_stat.runCycles_stat,
    //     commit_stat.idleCycles_stat, 
    //     commit_stat.blockCycles_fu_stat, 
    //     commit_stat.blockCycles_mem_stat, 
    //     commit_stat.blockCycles_depend_stat, 
    //     commit_stat.squashCycles_stat); // 57 + 6 = 63
#endif
}

void
DynInst::trap(const Fault &fault)
{
    cpu->trap(fault, threadNumber, staticInst);
}

Fault
DynInst::initiateMemRead(Addr addr, unsigned size, Request::Flags flags,
                               const std::vector<bool> &byte_enable)
{
    assert(byte_enable.size() == size);
    return cpu->pushRequest(
        dynamic_cast<DynInstPtr::PtrType>(this),
        /* ld */ true, nullptr, size, addr, flags, nullptr, nullptr,
        byte_enable);
}

Fault
DynInst::initiateMemMgmtCmd(Request::Flags flags)
{
    const unsigned int size = 8;
    return cpu->pushRequest(
            dynamic_cast<DynInstPtr::PtrType>(this),
            /* ld */ true, nullptr, size, 0x0ul, flags, nullptr, nullptr,
            std::vector<bool>(size, true));
}

Fault
DynInst::writeMem(uint8_t *data, unsigned size, Addr addr,
                        Request::Flags flags, uint64_t *res,
                        const std::vector<bool> &byte_enable)
{
    assert(byte_enable.size() == size);
    return cpu->pushRequest(
        dynamic_cast<DynInstPtr::PtrType>(this),
        /* st */ false, data, size, addr, flags, res, nullptr,
        byte_enable);
}

Fault
DynInst::initiateMemAMO(Addr addr, unsigned size, Request::Flags flags,
                              AtomicOpFunctorPtr amo_op)
{
    // atomic memory instructions do not have data to be written to memory yet
    // since the atomic operations will be executed directly in cache/memory.
    // Therefore, its `data` field is nullptr.
    // Atomic memory requests need to carry their `amo_op` fields to cache/
    // memory
    return cpu->pushRequest(
            dynamic_cast<DynInstPtr::PtrType>(this),
            /* atomic */ false, nullptr, size, addr, flags, nullptr,
            std::move(amo_op), std::vector<bool>(size, true));
}

} // namespace o3
} // namespace gem5
