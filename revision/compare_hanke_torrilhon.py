"""Item 15: channel-wise comparison of our reduced physical tensor R with the impact tensor S of
Hanke & Torrilhon (2023), Zenodo 10.5281/zenodo.7242835 (text file SmatrixAMX1099887766554433221100A,
Maxwell molecules, M = 20). Index map: their (n, a; nhat, b; ntilde, c) = our (l1, k1; l2, k2; l3, k3).
Their list stores each unordered pair {(b,nhat),(c,ntilde)} once with both orderings summed, so we
compare S against R^{(l1 l2 l3)}_{k1 k2 k3} + R^{(l1 l3 l2)}_{k1 k3 k2} (diagonal pairs once). Within a
channel (l1,l2,l3) the two must be proportional; the factor is the ratio of the angular normalizations
(coupling tensor vs. real Gaunt tensor) and is a symmetric function of the triple.
Usage: python3 compare_hanke_torrilhon.py <ours_R.csv> <K> <L>   (ours_R.csv: l1,l2,l3,k1,k2,k3,R)"""
import re, csv, collections, math, sys, os
HT=os.environ.get('HT_DIR','.')  # directory holding Hanke & Torrilhon's file SmatrixAMX1099887766554433221100A (Zenodo 10.5281/zenodo.7242835)
ours_csv, K, L = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
pat=re.compile(r'^\s*(\S+)\s+\d+\s+\d+\s+\d+\s+\d+\s+\\\\\s*NN=(\d+)\s+a=(\d+)\s+N1=(\d+)\s+b=(\d+)\s+N2=(\d+)\s+c=(\d+)')
ht={}
for line in open(os.path.join(HT,'SmatrixAMX1099887766554433221100A')):
    m=pat.match(line)
    if m: n,a,n1,b,n2,c=map(int,m.groups()[1:]); ht[(n,a,n1,b,n2,c)]=float(m.group(1))
R={}
for r in csv.DictReader(open(ours_csv)):
    R[(int(r['l1']),int(r['l2']),int(r['l3']),int(r['k1']),int(r['k2']),int(r['k3']))]=float(r['R'])
def ours(l1,k1,l2,k2,l3,k3):
    return R[(l1,l2,l3,k1,k2,k3)] if (l2,k2)==(l3,k3) else R[(l1,l2,l3,k1,k2,k3)]+R[(l1,l3,l2,k1,k3,k2)]
ch=collections.defaultdict(list)
for (n,a,n1,b,n2,c),v in ht.items():
    if max(n,n1,n2)<=L and max(a,b,c)<=K: ch[(n,n1,n2)].append((v,ours(n,a,n1,b,n2,c),(a,b,c)))
res=[]; A={}
for k,lst in sorted(ch.items()):
    so=sum(o*o for _,o,_ in lst); sv=sum(v*o for v,o,_ in lst); sv2=sum(v*v for v,_,_ in lst)
    a=sv/so; A[k]=a
    # residual computed directly (the expanded form sv2-2a sv+a^2 so cancels to the square root of machine epsilon)
    r=math.sqrt(sum((v-a*o)**2 for v,o,_ in lst)/sv2); res.append((r,k,len(lst)))
n_entries=sum(len(v) for v in ch.values()); n_multi=sum(1 for v in ch.values() if len(v)>1)
res.sort(reverse=True)
print(f"entries compared: {n_entries} in {len(ch)} channels ({n_multi} channels with >1 entry); l<={L}, k<={K}")
print(f"max relative residual {res[0][0]:.2e} (channel {res[0][1]}), median {sorted(r[0] for r in res)[len(res)//2]:.2e}")
print(f"max entry-wise relative deviation |v-a o|/max|v| over all channels: {max(abs(v-A[k]*o)/max(abs(x) for x,_,_ in lst) for k,lst in ch.items() for v,o,_ in lst):.2e}")
# symmetry of the per-channel factor under permutations of (l1,l2,l3)
sym=collections.defaultdict(set)
for k,a in A.items(): sym[tuple(sorted(k))].add(round(a,9))
print(f"per-channel factor: {len(A)} channels, {sum(1 for s in sym.values() if len(s)>1)} sorted triples with non-unique factor (expect 0)")
out=os.path.join(os.path.dirname(os.path.abspath(__file__)),'hanke_torrilhon_maxwell_comparison.csv')
with open(out,'w') as f:
    f.write('l1,l2,l3,n_entries,scale_factor,rel_residual\n')
    for r,k,n in sorted(res,key=lambda x:x[1]): f.write(f"{k[0]},{k[1]},{k[2]},{n},{A[k]:.10e},{r:.3e}\n")
print("written",out)
