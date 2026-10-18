import subprocess, numpy as np, sys
for src, dst in zip(sys.argv[1::2], sys.argv[2::2]):
    raw = subprocess.run(['ffmpeg','-loglevel','error','-i',src,'-f','s16le','-ac','1','-ar','22050','-'],capture_output=True).stdout
    x = np.frombuffer(raw,np.int16).astype(float); fr=441  # 20ms
    r = np.array([np.sqrt((x[i:i+fr]**2).mean())+1e-9 for i in range(0,len(x)-fr,fr)]); db=20*np.log10(r/32768)
    pk=int(np.argmax(db)); th=db[pk]-30
    s=pk
    while s>0 and db[s-1]>th: s-=1
    e=pk
    while e<len(db)-1 and db[e+1]>th: e+=1
    a=max(0,(s-1)*fr); b=min(len(x),(e+2)*fr)
    y=x[a:b].copy(); n=min(len(y),int(0.04*22050)); y[-n:]*=np.linspace(1,0,n)
    y=np.concatenate([y,np.zeros(int(0.05*22050))])
    subprocess.run(['ffmpeg','-y','-loglevel','error','-f','s16le','-ar','22050','-ac','1','-i','-','-b:a','32k',dst],input=y.astype(np.int16).tobytes(),check=True)
    print(src.split('/')[-1], round(len(x)/22050,2),'->',round(len(y)/22050,2))
