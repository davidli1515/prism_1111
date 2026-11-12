# Both summaries at 2 decimals on all files present now. Prints only, writes nothing.
import sys, os, re, types
sys.path.insert(0, os.getcwd())
def load(modname):
    src = open(modname + ".py").read()
    src = re.sub(r"(\d*)\.1%", lambda m: (m.group(1) or "") + ".2%", src)
    src = src.replace('sys.exit(f"exists, refusing to overwrite: {p}")', 'pass').replace('open(p, "w").write("\\n".join(L) + "\\n"); print("wrote", p)', 'pass')
    m = types.ModuleType(modname); m.__file__ = os.path.abspath(modname + ".py"); sys.modules[modname] = m
    exec(compile(src.replace('if __name__ == "__main__":', 'if False:'), m.__file__, "exec"), m.__dict__)
    return m
GV = load("gem5_val6_windowed"); GV.summary()
print("======")
TF = load("val6_trainfeat"); TF.GV = GV; TF.summary()
