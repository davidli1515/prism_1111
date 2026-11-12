#!/usr/bin/python
from optparse import OptionParser
import sys
import re
import json
import types
import math
import os
from xml.etree import ElementTree as ET


def parse(source):
    return ET.parse(source, parser=ET.XMLParser(target=ET.TreeBuilder()))

def main():
    global opts, outdir, tempFile
    usage = "usage: %prog [options] <gem5 stats file> <gem5 config file (json)> <mcpat template file>"
    parser = OptionParser(usage=usage)
    parser.add_option("-q", "--quiet", 
        action="store_false", dest="verbose", default=True,
        help="don't print status messages to stdout")
    parser.add_option("-o", "--out", type="string",
        action="store", dest="out", default="./",
        help="output directory (input to McPAT)")
    parser.add_option("-a", "--aggregate", action="store_true",
        help="aggregate all the stats in a periodic dump stat file")
    parser.add_option("-p", "--periodic", action="store_true",
        help="write each period in a separate xml file")
    (opts, args) = parser.parse_args()
    if len(args) != 3:
        parser.print_help()
        sys.exit(1)
    tempFile = args[2]
    outdir = opts.out
    if (outdir == "./"):
        outdir = os.path.dirname(args[0])
        outdir += "/mcpat-out"

    os.system("mkdir -p %s" %(outdir))
    readMcpatFile(args[2])
    readConfigFile(args[1])
    readStatsFile(args[0])
    #readConfigFile(args[1])
    #readMcpatFile(args[2])
    #os.system("mkdir -p %s" %(outdir))
    #dumpMcpatOut(outdir, args[2])

def dumpMcpatOut(itr):
    outDir = outdir
    templateFile = tempFile
    rootElem = templateMcpat.getroot()
    configMatch = re.compile(r'config\.([a-zA-Z0-9_:\.]+)')
    #replace params with values from the GEM5 config file 
    for param in rootElem.iter('param'):
        name = param.attrib['name']
        value = param.attrib['value']
        print(name)
        print(value)
        if 'config' in value:
            allConfs = configMatch.findall(value)
            for conf in allConfs:
                confValue = getConfValue(conf)
                value = re.sub("config."+ conf, str(confValue), value)
            try:
                if "," in value:
                    exprs = re.split(',', value)
                    for i in range(len(exprs)):
                        exprs[i] = str(eval(exprs[i]))
                    param.attrib['value'] = ','.join(exprs)
                elif "[" in value or "]" in value:
                    value_proc = str(value).replace('[', '')
                    value_proc = str(value_proc).replace(']', '')
                    param.attrib['value'] = str(eval(str(value_proc)))
                else:
                    param.attrib['value'] = str(eval(str(value)))
            except Exception as e:
                print("WARNING: Failed to eval param '%s' with value '%s': %s" % (name, value, e))
                param.attrib['value'] = "0"

    #replace stats with values from the GEM5 stats file 
    statRe = re.compile(r'stats\.([a-zA-Z0-9_:\.]+)')
    for j in range(period + 1):
        for stat in rootElem.iter('stat'):
            name = stat.attrib['name']
            value = stat.attrib['value']
            if 'stats' in value:
                allStats = statRe.findall(value)
                expr = value
                for i in range(len(allStats)):
                    if allStats[i] in stats[j]:
                        expr = re.sub('stats.%s' % allStats[i], stats[j][allStats[i]], expr)
                    else:
                        expr = re.sub('stats.%s' % allStats[i], "0.0", expr)
                        #print "***WARNING: %s does not exist in stats***" % allStats[i]
                        #print "\t Please use the right stats in your McPAT template file"

                if 'config' not in expr and 'stats' not in expr:
                    if "/ 0.0" in expr:
                        stat.attrib['value'] = str(0)
                    else:
                        try:
                            stat.attrib['value'] = str(eval(expr))
                        except Exception as e:
                            print("WARNING: Failed to eval stat '%s' with expr '%s': %s" % (name, expr, e))
                            stat.attrib['value'] = "0"
        #Write out the xml file
        if opts.verbose: print("Writing input to McPAT in: %s" % outDir) 
        templateMcpat.write("%s/mcpat-out-%d.xml" %(outDir, itr))
        #if (itr == 0):
        #    os.system("sed -i 's/<stat name=\"clock_rate\" value=\"[0-9]*\"/<param name=\"clock_rate\" value=\"3100\"/g' %s/mcpat-out-%d.xml" %(outDir, itr))
        #    os.system("sed -i 's/<stat name=\"vdd\" value=\"[01]\.[0-9]*\"/<param name=\"vdd\" value=\"1.225\"/g' %s/mcpat-out-%d.xml" %(outDir, itr))
        #else:
        os.system("sed -i 's/<stat name=\"clock_rate_dvfs\"/<param name=\"clock_rate_dvfs\"/g' %s/mcpat-out-%d.xml" %(outDir, itr))
        os.system("sed -i 's/<stat name=\"vdd_dvfs\"/<param name=\"vdd_dvfs\"/g' %s/mcpat-out-%d.xml" %(outDir, itr))
        os.system("sed -i 's/stat name=\"sim_second/param name=\"sim_second/g' %s/mcpat-out-%d.xml" %(outDir, itr))
        os.system("sed -i 's/stat name=\"sim_ticks/param name=\"sim_ticks/g' %s/mcpat-out-%d.xml" %(outDir, itr))
        readMcpatFile(templateFile)
        rootElem = templateMcpat.getroot()
        configMatch = re.compile(r'config\.([a-zA-Z0-9_:\.]+)')

        for param in rootElem.iter('param'):
            name = param.attrib['name']
            value = param.attrib['value']
            if 'config' in value:
                allConfs = configMatch.findall(value)
                for conf in allConfs:
                    confValue = getConfValue(conf)
                    value = re.sub("config."+ conf, str(confValue), value)
                try:
                    if "," in value:
                        exprs = re.split(',', value)
                        for i in range(len(exprs)):
                            exprs[i] = str(eval(exprs[i]))
                        param.attrib['value'] = ','.join(exprs)
                        # prevent eval() from making tuple in else
                        continue
                    if "[" in value or "]" in value:
                        value_proc = str(value).replace('[', '')
                        value_proc = str(value_proc).replace(']', '')
                        param.attrib['value'] = str(eval(str(value_proc)))
                    else:
                        param.attrib['value'] = str(eval(str(value)))
                except Exception as e:
                    print("WARNING: Failed to eval param '%s' with value '%s': %s" % (name, value, e))
                    param.attrib['value'] = "0"


def getConfValue(confStr):
    # Path aliases for known GEM5 version differences
    # Template may use old paths that don't exist in newer GEM5 configs
    PATH_ALIASES = {
        'branchPred.ras.numEntries': 'branchPred.RASSize',
        'branchPred.BTBEntries': 'branchPred.BTBEntries',
    }

    # Apply path aliases if the original path fails
    result = _tryGetConfValue(confStr)
    if result == "__NOT_FOUND__":
        for old_suffix, new_suffix in PATH_ALIASES.items():
            if old_suffix in confStr:
                alt = confStr.replace(old_suffix, new_suffix)
                print("INFO: Trying path alias '%s' -> '%s'" % (confStr, alt))
                result = _tryGetConfValue(alt)
                if result != "__NOT_FOUND__":
                    return result
    if result == "__NOT_FOUND__":
        print("***WARNING: %s not found anywhere in config***" % confStr)
        return "0"
    return result

def _tryGetConfValue(confStr):
    """Try confStr with cpu, then switch_cpus variants."""
    # GEM5 warmup/fast-forward redirect:
    # system.cpu = warmup (AtomicSimpleCPU)
    # system.switch_cpus = TimingSimpleCPU
    # system.switch_cpus_1 = detailed O3CPU
    result = _getConfValueInner(confStr)
    if result == "__NOT_FOUND__" and '.cpu.' in confStr:
        # Try switch_cpus variants - O3CPU (switch_cpus_1) first
        for switch_name in ['switch_cpus_1', 'switch_cpus_2', 'switch_cpus']:
            alt = confStr.replace('.cpu.', '.%s.' % switch_name)
            print("INFO: '%s' not found, trying '%s'" % (confStr, alt))
            result = _getConfValueInner(alt)
            if result != "__NOT_FOUND__":
                return result
    return result

def _getConfValueInner(confStr):
    spltConf = re.split('\.', confStr) 
    currConf = config
    currHierarchy = ""
    for idx, x in enumerate(spltConf):
        currHierarchy += x
        if currConf is None:
            return "__NOT_FOUND__"
        elif isinstance(currConf, str) and idx < len(spltConf):
            # currConf is a string reference (e.g., "system.cpu_clk_domain")
            # Follow the reference: resolve it from config root, then continue
            # with remaining path keys including current x
            remaining = [x] + spltConf[idx+1:]
            ref_path = currConf.split('.')
            resolved = config
            for r in ref_path:
                if isinstance(resolved, dict) and r in resolved:
                    resolved = resolved[r]
                elif isinstance(resolved, list) and len(resolved) > 0:
                    resolved = resolved[0].get(r) if isinstance(resolved[0], dict) else None
                else:
                    return "__NOT_FOUND__"
            # Now continue traversing remaining keys from resolved
            for rem in remaining:
                if resolved is None:
                    return "__NOT_FOUND__"
                elif isinstance(resolved, dict) and rem in resolved:
                    resolved = resolved[rem]
                elif isinstance(resolved, list) and len(resolved) > 0:
                    if isinstance(resolved[0], dict) and rem in resolved[0]:
                        resolved = resolved[0][rem]
                    else:
                        return "__NOT_FOUND__"
                else:
                    return "__NOT_FOUND__"
            # Handle single-element lists
            if isinstance(resolved, list) and len(resolved) == 1:
                resolved = resolved[0]
            if isinstance(resolved, (dict, list)):
                return "0"
            return resolved
        elif not isinstance(currConf, (dict, list)):
            # currConf is a scalar (int, etc.) but we still have keys to traverse
            return "__NOT_FOUND__"
        elif isinstance(currConf, list):
            # system.cpu is an array - try to find a non-Atomic (detailed) CPU
            found = False
            for cpu_entry in currConf:
                if isinstance(cpu_entry, dict) and x in cpu_entry:
                    if 'Atomic' not in cpu_entry.get('type', ''):
                        currConf = cpu_entry[x]
                        found = True
                        break
            if not found:
                for cpu_entry in currConf:
                    if isinstance(cpu_entry, dict) and x in cpu_entry:
                        currConf = cpu_entry[x]
                        found = True
                        break
            if not found:
                return "__NOT_FOUND__"
        elif x not in currConf:
            return "__NOT_FOUND__"
        else:
            currConf = currConf[x]
            if currHierarchy == "testsys.cpu_clk_domain.clock":
                currConf = currConf[0] / 1000000000000.0
        currHierarchy += "."
    # Handle single-element lists (e.g., clock: [500])
    if isinstance(currConf, list) and len(currConf) == 1:
        currConf = currConf[0]
    # Ensure we return a scalar, not a complex object
    if isinstance(currConf, (dict, list)):
        print("WARNING: config value for '%s' is a complex object, returning 0" % confStr)
        return "0"
    return currConf
    

def readStatsFile(statsFile):
    global stats, period
    stats = {}
    period = 0
    if opts.verbose: print("Reading GEM5 stats from: %s" %  statsFile)
    F = open(statsFile)
    ignores = re.compile(r'^---|^$')
    statLine = re.compile(r'([a-zA-Z0-9_\.:+-]+)\s+([-+]?[0-9]+\.[0-9]+|[-+]?[0-9]+|nan|inf)')
    count = 0
    itr = 0
    for line in F:
        #ignore empty lines and lines starting with "---"  
        if not ignores.match(line):
            #if "work_item" in line:
            #    continue
            #print line
            m = statLine.match(line)
            if m is None:
                continue
            count += 1
            statKind = m.group(1)
            statValue = m.group(2)
            if statValue == 'nan':
                #print "\tWarning (stats): %s is nan. Setting it to 0" % statKind
                statValue = '0.0'
            if period not in stats:
                stats[period] = {}
            if not opts.periodic and statKind in stats[period]:
                stats[period][statKind] = str(float(stats[period][statKind]) + float(statValue))
            else:
                stats[period][statKind] = statValue

        if "End Simulation Statistics" in line:
            if not opts.aggregate:
                dumpMcpatOut(itr)
                break
            if opts.periodic:
                dumpMcpatOut(itr)
                itr += 1
                #period += 1
    if not opts.periodic:
        dumpMcpatOut(itr)
    F.close()

def readConfigFile(configFile):
    global config
    if opts.verbose: print("Reading config from: %s" % configFile)
    F = open(configFile)
    config = json.load(F)
    #print(config)
    #print config["system"]["membus"]
    #print config["system"]["cpu"][0]["clock"]
    F.close()

def readMcpatFile(templateFile):
    global templateMcpat 
    if opts.verbose: print("Reading McPAT template from: %s" % templateFile)
    templateMcpat = parse(templateFile)
    #print dir(templateMcpat)
    
    

if __name__ == '__main__':
    main()