import random as r,math as h
r.seed(42)
d=[i.strip()for i in open("input.txt").read().strip().split("\n")if i.strip()];r.shuffle(d);u=sorted(set(''.join(d)));b=len(u)
print(f"num docs: {len(d)}");print(f"vocab size: {b+1}")
R=0,1,2,3
class V:
 def __init__(s,d,c=(),l=()):s.d=d;s.g=0;s.c=c;s.l=l
 def __add__(s,o):o=o if type(o)==V else V(o);return V(s.d+o.d,(s,o),(1,1))
 __radd__=__add__
 def __mul__(s,o):o=o if type(o)==V else V(o);return V(s.d*o.d,(s,o),(o.d,s.d))
 def __truediv__(s,o):return s*o**-1
 def __pow__(s,o):return V(s.d**o,(s,),(o*s.d**(o-1),))
 def log(s):return V(h.log(s.d),(s,),(1/s.d,))
 def exp(s):e=h.exp(s.d);return V(e,(s,),(e,))
 def relu(s):return V(max(0,s.d),(s,),(s.d>0,))
 def backward(s):
  t=[];v=set()
  def f(x):
   if x not in v:v.add(x);[f(i)for i in x.c];t.append(x)
  f(s);s.g=1
  for x in t[::-1]:
   for i,j in zip(x.c,x.l):i.g+=j*x.g
M=lambda o,i:[[V(r.gauss(0,.08))for _ in range(i)]for _ in range(o)]
e,p,o,q,k,v,w,f,g=M(b+1,16),M(16,16),M(b+1,16),M(16,16),M(16,16),M(16,16),M(16,16),M(64,16),M(16,64)
P=[p for m in(e,p,o,q,k,v,w,f,g)for r in m for p in r]
print(f"num params: {len(P)}")
L=lambda x,w:[sum(a*b for a,b in zip(i,x))for i in w]
def S(z):
 x=max(i.d for i in z);e=[(i+x*-1).exp()for i in z];q=sum(e);return[i/q for i in e]
def N(x):
 y=sum(i*i for i in x)/len(x);y=(y+1e-5)**-.5;return[i*y for i in x]
def G(t,z,a,c):
 x=N([i+j for i,j in zip(e[t],p[z])]);u0=x;x=N(x);j=L(x,q);l=L(x,k);n=L(x,v);a+=[l];c+=[n];h=[]
 for i in 0,4,8,12:
  y0=j[i:i+4];l=[s[i:i+4]for s in a];n=[s[i:i+4]for s in c];t=[sum(y0[j]*l[k][j]for j in R)/2 for k in range(len(l))];t=S(t);h+=[sum(t[k]*n[k][j]for k in range(len(n)))for j in R]
 x=[i+j for i,j in zip(L(h,w),u0)];u0=x;x=L(N(x),f);x=[i.relu()for i in x];x=[i+j for i,j in zip(L(x,g),u0)]
 return L(x,o)
m=[0.]*4192;n=[0.]*4192
for i in range(1000):
 t=[b]+[u.index(j)for j in d[i%len(d)]]+[b];z=min(16,len(t)-1);a=[];c=[];x=[]
 for j in range(z):x+=[S(G(t[j],j,a,c))[t[j+1]].log()*-1]
 x=sum(x)/z;x.backward();l=.01*(1-i*.001)
 for j,z0 in enumerate(P):
  m[j]=.85*m[j]+.15*z0.g;n[j]=.99*n[j]+.01*z0.g**2;u1=m[j]/(1-.85**(i+1));u2=n[j]/(1-.99**(i+1));z0.d-=l*u1/(u2**.5+1e-8);z0.g=0
 print(f"step {i+1:4d} / {1000:4d} | loss {x.d:.4f}")
print("\n--- inference (new, hallucinated names) ---")
for i in range(20):
 a=[];c=[];t=b;s=[]
 for z in range(16):
  t=r.choices(range(b+1),weights=[i.d for i in S([j/.5 for j in G(t,z,a,c)])])[0]
  if t==b:break
  s.append(u[t])
 print(f"sample {i+1:2d}: {''.join(s)}")
