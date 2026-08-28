"""Lean build babysitter — sequentialise thrashing lean workers without killing anything.

The box (16 GB) cannot hold N concurrent lean kernel-scan slices; they page-thrash to
~2-10% CPU each and never finish. This loop keeps at most MAX_RUN lean.exe processes
running and suspends the rest (fully reversible; no work lost — suspended state lives in
the pagefile, finished slices write .olean artifacts that lake's next invocation reuses).
As a running worker exits (slice done), the longest-suspended worker is resumed. New
workers spawned by the still-alive lake supervisors are folded into the same policy.

Run:    C:\dev\primes\.venv\Scripts\python.exe research\lean_babysitter.py
Stops itself when no lean.exe remains. Log: research/data/lean_babysitter.log
Priority: keeps the worker with the largest resident set running first (it is closest
to being memory-warm); ties broken by earliest start time (most CPU invested).
"""
import psutil, time, sys, io, ctypes

_psapi = ctypes.WinDLL("psapi")
_kernel = ctypes.WinDLL("kernel32")
PROCESS_SET_QUOTA = 0x0100 | 0x0400  # + PROCESS_QUERY_INFORMATION, required by EmptyWorkingSet
def trim(pid):
    # Empty a (suspended) process's working set: pages move to pagefile NOW,
    # freeing physical RAM for the running worker and the UI. Safe for
    # suspended processes - they fault pages back in when resumed.
    h = _kernel.OpenProcess(PROCESS_SET_QUOTA, False, pid)
    if not h:
        return False
    try:
        return bool(_psapi.EmptyWorkingSet(h))
    finally:
        _kernel.CloseHandle(h)

MAX_RUN = 2
LOW_MEM = 1.5 * 2**30    # below this available RAM: run at most 1
CRIT_MEM = 0.75 * 2**30  # below this: suspend everything until recovery
RECOVER = 2.0 * 2**30    # resume normal policy above this
LOG = r"C:\dev\primes\research\data\lean_babysitter.log"

def log(msg):
    line = time.strftime("%H:%M:%S ") + msg
    print(line, flush=True)
    with io.open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")

def leans():
    out = []
    for p in psutil.process_iter(["name"]):
        try:
            if p.info["name"] and p.info["name"].lower() == "lean.exe":
                out.append(p)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return out

log(f"babysitter start, MAX_RUN={MAX_RUN}")
suspended_at = {}  # pid -> time we suspended it (for FIFO resume)
idle_polls = 0
while True:
    procs = leans()
    if not procs:
        idle_polls += 1
        if idle_polls >= 3:
            log("no lean processes remain - resuming any stragglers and exiting")
            break
        time.sleep(20)
        continue
    idle_polls = 0
    # memory guard: the box must never hang again. Scale allowed runners by free RAM.
    avail = psutil.virtual_memory().available
    if avail < CRIT_MEM:
        allowed = 0
    elif avail < LOW_MEM:
        allowed = 1
    elif avail >= RECOVER:
        allowed = MAX_RUN
    else:
        allowed = 1  # between LOW and RECOVER: stay conservative
    if allowed != MAX_RUN:
        log(f"mem guard: avail={avail//2**20}MB -> allowed={allowed}")
    running, susp = [], []
    for p in procs:
        try:
            (susp if p.status() == psutil.STATUS_STOPPED else running).append(p)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    # rank running: largest resident set first, then oldest
    def rank(p):
        try:
            return (-p.memory_info().rss, p.create_time())
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return (0, 0)
    running.sort(key=rank)
    # too many running -> suspend the tail
    for p in running[allowed:]:
        try:
            p.suspend()
            suspended_at[p.pid] = time.time()
            ok = trim(p.pid)
            log(f"suspend pid={p.pid} rss={p.memory_info().rss//2**20}MB trim={'ok' if ok else 'FAIL'}")
        except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
            log(f"suspend pid={p.pid} failed: {e}")
    # keep suspended workers trimmed (Windows can lazily fault pages back)
    for p in susp:
        try:
            if p.memory_info().rss > 64 * 2**20:
                trim(p.pid)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    # free slots -> resume longest-suspended first
    slots = allowed - min(len(running), allowed)
    if slots > 0 and susp:
        susp.sort(key=lambda p: suspended_at.get(p.pid, 0))
        for p in susp[:slots]:
            try:
                p.resume()
                log(f"resume pid={p.pid}")
                suspended_at.pop(p.pid, None)
            except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                log(f"resume pid={p.pid} failed: {e}")
    time.sleep(60)

# safety: never exit leaving anything suspended
for p in leans():
    try:
        if p.status() == psutil.STATUS_STOPPED:
            p.resume()
            log(f"exit-resume pid={p.pid}")
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass
log("babysitter done")
