"""Round 95 / loop entry 101: how the prime-compatible killer fraction falls with p.

Exact counting stops near p = 31.  This samples prime-compatible residue vectors at random for the
narrowest stretches (twin gears p, p+2, where the stretch is about 2p/3 columns and the adversary
has the easiest job) and estimates the fraction that cover the stretch.  If the fraction decays
faster than the number of stretches grows, the heuristic expectation of a dead stretch anywhere is
dominated by the small p already checked exactly.  A heuristic, recorded as one.
"""
import random, sys, math

def primes_to(n):
    s=bytearray([1])*(n+1); s[0:2]=b"\x00\x00"
    for i in range(2,int(n**0.5)+1):
        if s[i]: s[i*i::i]=bytearray(len(s[i*i::i]))
    return [i for i in range(2,n+1) if s[i]]

def main(argv):
    N=int(argv[0]) if argv else 200000
    rng=random.Random(11)
    ps=primes_to(400)
    twins=[(p,p+2) for p in ps if p>=17 and (p+2) in set(ps)]
    print("sampled prime-compatible residue vectors at twin-gear stretches (p^2, (p+2)^2], %d samples each"%N)
    print("     p   columns   covers found   fraction (or bound)   -ln(fraction) / (p / ln^2 p)")
    for p,q in twins:
        if p>200: break
        gears=[h for h in ps if 5<=h<=p]
        L=(q*q-p*p)//6
        full=((1<<L)-1)&~(1<<(L-1))
        maskof={}
        for h in gears:
            inv6=pow(6,-1,h); per=[]
            for r in range(h):
                c=((r*r+5)*inv6)%h; m=0
                for tooth in (inv6%h,(-inv6)%h):
                    t=(tooth-c)%h
                    while t<L: m|=1<<t; t+=h
                per.append(m)
            maskof[h]=per
        hits=0
        for _ in range(N):
            cov=maskof[p][0]
            for h in gears[:-1]:
                cov|=maskof[h][rng.randrange(1,h)]
                if cov&full==full: break
            if cov&full==full: hits+=1
        frac=hits/N
        scale=p/(math.log(p)**2)
        if hits:
            print("  %4d  %8d  %12d   %10.2e            %6.3f"%(p,L,hits,frac,-math.log(frac)/scale))
        else:
            print("  %4d  %8d  %12d   < %8.1e            > %5.3f"%(p,L,hits,1.0/N,-math.log(1.0/N)/scale))

if __name__=="__main__":
    main(sys.argv[1:])
