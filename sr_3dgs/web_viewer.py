"""Generate a simple, WORKING WebGL viewer for .splat files.
Uses vertex duplication (4 vertices per Gaussian) instead of instanced rendering.
This is the simplest possible approach that works everywhere."""

import base64
from pathlib import Path

def _size_fmt(b):
    for u in ["B","KB","MB","GB"]:
        if b < 1024: return f"{b:.1f} {u}"
        b /= 1024

def generate_viewer(splat_path, output_html, title="3D View", embed_splat=True, max_splat_mb=15, background_color="#1a1a2e"):
    splat_file = Path(splat_path)
    if not splat_file.exists():
        raise FileNotFoundError(f".splat not found: {splat_path}")
    sz = splat_file.stat().st_size

    if embed_splat and sz < max_splat_mb * 1024 * 1024:
        with open(splat_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
        print(f"[Viewer] Embedding {_size_fmt(sz)} base64")
        dc = f'var SPLAT_B64="{b64}";var SPLAT_URL=null;'
    else:
        print(f"[Viewer] External ref: {splat_file.name} ({_size_fmt(sz)})")
        dc = f'var SPLAT_B64=null;var SPLAT_URL="{splat_file.name}";'

    html = _TEMPLATE.replace("{{TITLE}}", title)
    html = html.replace("{{DATA_CONFIG}}", dc)
    html = html.replace("{{BACKGROUND}}", background_color)
    html = html.replace("{{FILE_INFO}}", f"{splat_file.name} - {_size_fmt(sz)}")

    with open(output_html, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[Viewer] {output_html}")

_TEMPLATE = '''<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0,maximum-scale=1.0,user-scalable=no"><title>{{TITLE}}</title><style>*{margin:0;padding:0;box-sizing:border-box}html,body{width:100%;height:100%;overflow:hidden;background:{{BACKGROUND}};font-family:Arial,sans-serif;touch-action:none}canvas{display:block;position:fixed;top:0;left:0}#loading{position:fixed;top:0;left:0;width:100%;height:100%;display:flex;flex-direction:column;align-items:center;justify-content:center;background:{{BACKGROUND}};z-index:100;color:#fff;font-size:16px}#info{position:fixed;bottom:10px;left:50%;transform:translateX(-50%);color:rgba(255,255,255,0.5);font-size:11px;pointer-events:none;z-index:10}</style></head><body><canvas id="c"></canvas><div id="loading"><div id="msg">Loading 3D Scene...</div></div><div id="info">{{FILE_INFO}}</div><script>{{DATA_CONFIG}}
var canvas=document.getElementById("c");
var msg=document.getElementById("msg");
var gl=canvas.getContext("webgl",{antialias:false,alpha:false,preserveDrawingBuffer:false})||canvas.getContext("experimental-webgl",{antialias:false,alpha:false,preserveDrawingBuffer:false});
if(!gl){msg.textContent="WebGL not supported";throw new Error("no webgl");}

var vs=gl.createShader(gl.VERTEX_SHADER);
gl.shaderSource(vs,"attribute vec3 aPos;attribute vec3 aScale;attribute vec4 aColor;attribute vec4 aRot;attribute vec2 aCorner;uniform mat4 uProj;uniform mat4 uView;uniform vec2 uFocal;uniform vec2 uVP;varying vec4 vColor;varying vec2 vPos;vec3 qrot(vec4 q,vec3 v){vec4 qq=normalize(q);vec3 t=2.0*cross(qq.yzw,v);return v+qq.x*t+cross(qq.yzw,t);}void main(){vec4 cp=uView*vec4(aPos,1.0);vec4 pp=uProj*cp;float s=max(0.001,abs(cp.z));vec3 ca=qrot(aRot,vec3(aScale.x,0.0,0.0));vec3 cb=qrot(aRot,vec3(0.0,aScale.y,0.0));vec3 J1=vec3(uFocal.x/s,0.0,-(cp.x*uFocal.x)/(s*s));vec3 J2=vec3(0.0,uFocal.y/s,-(cp.y*uFocal.y)/(s*s));float xx=dot(J1,ca)*dot(J1,ca)+dot(J1,cb)*dot(J1,cb);float yy=dot(J2,ca)*dot(J2,ca)+dot(J2,cb)*dot(J2,cb);float xy=dot(J1,ca)*dot(J2,ca)+dot(J1,cb)*dot(J2,cb);float r=3.0;float disc=sqrt(max(0.0,(xx-yy)*(xx-yy)+4.0*xy*xy));float e1=0.5*(xx+yy+disc);float e2=0.5*(xx+yy-disc);vec2 ev=normalize(vec2(xy,e1-xx));if(length(ev)<0.001)ev=vec2(1.0,0.0);float w=r*sqrt(max(0.0,e1));float h=r*sqrt(max(0.0,e2));vec2 off=(ev*w*aCorner.x+vec2(-ev.y,ev.x)*h*aCorner.y);vec2 ndc=pp.xy/pp.w;gl_Position=vec4(ndc+off/uVP*2.0,pp.z/pp.w,1.0);vColor=aColor;vPos=aCorner;}");
gl.compileShader(vs);
if(!gl.getShaderParameter(vs,gl.COMPILE_STATUS)){msg.textContent="VS error:"+gl.getShaderInfoLog(vs);throw new Error("vs");}

var fs=gl.createShader(gl.FRAGMENT_SHADER);
gl.shaderSource(fs,"precision highp float;varying vec4 vColor;varying vec2 vPos;void main(){float a=vColor.a*exp(-0.5*dot(vPos,vPos));if(a<0.0039)discard;gl_FragColor=vec4(vColor.rgb,a);}");
gl.compileShader(fs);
if(!gl.getShaderParameter(fs,gl.COMPILE_STATUS)){msg.textContent="FS error:"+gl.getShaderInfoLog(fs);throw new Error("fs");}

var prog=gl.createProgram();
gl.attachShader(prog,vs);gl.attachShader(prog,fs);gl.linkProgram(prog);
if(!gl.getProgramParameter(prog,gl.LINK_STATUS)){msg.textContent="Link error:"+gl.getProgramInfoLog(prog);throw new Error("link");}
gl.useProgram(prog);

var uProj=gl.getUniformLocation(prog,"uProj");
var uView=gl.getUniformLocation(prog,"uView");
var uFocal=gl.getUniformLocation(prog,"uFocal");
var uVP=gl.getUniformLocation(prog,"uVP");

var CORNERS=new Float32Array([-1,-1,1,-1,1,1,-1,-1,1,1,-1,1]);

function createBuffer(data,loc,size,stride,offset){
  var buf=gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER,buf);
  gl.bufferData(gl.ARRAY_BUFFER,data,gl.STATIC_DRAW);
  gl.enableVertexAttribArray(loc);
  gl.vertexAttribPointer(loc,size,gl.FLOAT,false,stride||0,offset||0);
}

var N=0,cx=0,cy=0,cz=0;
var camDist=3,camTheta=0,camPhi=Math.PI/4;
var target=[0,0,0],fov=60;
var touching=false,lx=0,ly=0,ld=0,vx=0,vy=0;

function getEye(){
  return [target[0]+camDist*Math.sin(camPhi)*Math.cos(camTheta),
          target[1]+camDist*Math.sin(camPhi)*Math.sin(camTheta),
          target[2]+camDist*Math.cos(camPhi)];
}

function makeView(eye,tgt,up){
  var z=[eye[0]-tgt[0],eye[1]-tgt[1],eye[2]-tgt[2]];
  var zl=Math.sqrt(z[0]*z[0]+z[1]*z[1]+z[2]*z[2]);z[0]/=zl;z[1]/=zl;z[2]/=zl;
  var x=[up[1]*z[2]-up[2]*z[1],up[2]*z[0]-up[0]*z[2],up[0]*z[1]-up[1]*z[0]];
  var xl=Math.sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2]);x[0]/=xl;x[1]/=xl;x[2]/=xl;
  var y=[z[1]*x[2]-z[2]*x[1],z[2]*x[0]-z[0]*x[2],z[0]*x[1]-z[1]*x[0]];
  return new Float32Array([x[0],y[0],z[0],0,x[1],y[1],z[1],0,x[2],y[2],z[2],0,
    -(x[0]*eye[0]+x[1]*eye[1]+x[2]*eye[2]),
    -(y[0]*eye[0]+y[1]*eye[1]+y[2]*eye[2]),
    -(z[0]*eye[0]+z[1]*eye[1]+z[2]*eye[2]),1]);
}

function makeProj(fovY,aspect,near,far){
  var f=1.0/Math.tan(fovY*Math.PI/360.0);
  var ri=1.0/(near-far);
  return new Float32Array([f/aspect,0,0,0,0,f,0,0,0,0,(near+far)*ri,-1,0,0,near*far*ri*2,0]);
}

function loadSplatData(buffer){
  N=Math.floor(buffer.byteLength/32);
  msg.textContent="Parsing "+N.toLocaleString()+" points...";

  var V=N*6;
  var posArr=new Float32Array(V*3);
  var scaleArr=new Float32Array(V*3);
  var colorArr=new Float32Array(V*4);
  var rotArr=new Float32Array(V*4);

  var u8=new Uint8Array(buffer);
  var view=new DataView(buffer);
  var sx=0,sy=0,sz=0;

  for(var i=0;i<N;i++){
    var off=i*32;
    var px=view.getFloat32(off,true);
    var py=view.getFloat32(off+4,true);
    var pz=view.getFloat32(off+8,true);
    sx+=px;sy+=py;sz+=pz;

    var scx=view.getFloat32(off+12,true);
    var scy=view.getFloat32(off+16,true);
    var scz=view.getFloat32(off+20,true);

    var cr=u8[off+24]/255.0;
    var cg=u8[off+25]/255.0;
    var cb=u8[off+26]/255.0;
    var ca=u8[off+27]/255.0;

    var r0=(u8[off+28]/255.0)*2.0-1.0;
    var r1=(u8[off+29]/255.0)*2.0-1.0;
    var r2=(u8[off+30]/255.0)*2.0-1.0;
    var r3=(u8[off+31]/255.0)*2.0-1.0;

    for(var j=0;j<6;j++){
      var vi=(i*6+j);
      posArr[vi*3]=px;posArr[vi*3+1]=py;posArr[vi*3+2]=pz;
      scaleArr[vi*3]=scx;scaleArr[vi*3+1]=scy;scaleArr[vi*3+2]=scz;
      colorArr[vi*4]=cr;colorArr[vi*4+1]=cg;colorArr[vi*4+2]=cb;colorArr[vi*4+3]=ca;
      rotArr[vi*4]=r0;rotArr[vi*4+1]=r1;rotArr[vi*4+2]=r2;rotArr[vi*4+3]=r3;
    }
  }

  cx=sx/N;cy=sy/N;cz=sz/N;
  var dists=new Float32Array(N),maxDist=0,px,py,pz,d;
  for(var i=0;i<N;i++){
    var off=i*32;
    px=view.getFloat32(off,true)-cx;
    py=view.getFloat32(off+4,true)-cy;
    pz=view.getFloat32(off+8,true)-cz;
    d=Math.sqrt(px*px+py*py+pz*pz);
    dists[i]=d;
    if(d>maxDist)maxDist=d;
  }
  dists.sort();
  var p95=dists[Math.min(N-1,Math.floor(N*0.95))];
  target=[cx,cy,cz];
  camDist=Math.max(2,p95*3.0);
  origCamDist=camDist;

  createBuffer(posArr,gl.getAttribLocation(prog,"aPos"),3,0,0);
  createBuffer(scaleArr,gl.getAttribLocation(prog,"aScale"),3,0,0);
  createBuffer(colorArr,gl.getAttribLocation(prog,"aColor"),4,0,0);
  createBuffer(rotArr,gl.getAttribLocation(prog,"aRot"),4,0,0);

  var cornerBuf=new Float32Array(V*2);
  for(var i=0;i<N;i++){
    for(var j=0;j<6;j++){
      cornerBuf[(i*6+j)*2]=CORNERS[j*2];
      cornerBuf[(i*6+j)*2+1]=CORNERS[j*2+1];
    }
  }
  createBuffer(cornerBuf,gl.getAttribLocation(prog,"aCorner"),2,0,0);

  msg.textContent=N.toLocaleString()+" points loaded";
  setTimeout(function(){
    document.getElementById("loading").style.display="none";
  },500);
}

function resize(){
  canvas.width=window.innerWidth*(window.devicePixelRatio||1);
  canvas.height=window.innerHeight*(window.devicePixelRatio||1);
  gl.viewport(0,0,canvas.width,canvas.height);
  gl.uniform2f(uVP,canvas.width,canvas.height);
}
window.addEventListener("resize",resize);
resize();

function render(){
  if(N===0){requestAnimationFrame(render);return;}
  if(!touching){camTheta+=vx*0.016;camPhi+=vy*0.016;vx*=0.92;vy*=0.92;}
  camPhi=Math.max(0.1,Math.min(Math.PI-0.1,camPhi));
  var eye=getEye();
  var vmat=makeView(eye,target,[0,0,1]);
  var aspect=canvas.width/Math.max(1,canvas.height);
  var pmat=makeProj(fov,aspect,0.1,500.0);
  var fx=canvas.width/(2.0*Math.tan(fov*Math.PI/360.0));
  var fy=canvas.height/(2.0*Math.tan(fov*Math.PI/360.0));
  gl.uniformMatrix4fv(uProj,false,pmat);
  gl.uniformMatrix4fv(uView,false,vmat);
  gl.uniform2f(uFocal,fx,fy);
  gl.uniform2f(uVP,canvas.width,canvas.height);
  gl.clearColor(0.1,0.1,0.18,1.0);
  gl.clear(gl.COLOR_BUFFER_BIT);
  gl.enable(gl.BLEND);
  gl.blendFuncSeparate(gl.ONE_MINUS_DST_ALPHA,gl.ONE,gl.ONE_MINUS_DST_ALPHA,gl.ONE);
  gl.blendEquationSeparate(gl.FUNC_ADD,gl.FUNC_ADD);
  gl.drawArrays(gl.TRIANGLES,0,6*N);
  requestAnimationFrame(render);
}

canvas.addEventListener("touchstart",function(e){e.preventDefault();touching=true;vx=0;vy=0;if(e.touches.length===1){lx=e.touches[0].clientX;ly=e.touches[0].clientY;}else if(e.touches.length===2){var dx=e.touches[0].clientX-e.touches[1].clientX;var dy=e.touches[0].clientY-e.touches[1].clientY;ld=Math.sqrt(dx*dx+dy*dy);}},{passive:false});
canvas.addEventListener("touchmove",function(e){e.preventDefault();if(e.touches.length===1){var dx=e.touches[0].clientX-lx;var dy=e.touches[0].clientY-ly;camTheta-=dx*0.005;camPhi+=dy*0.005;vx=dx*0.005*60;vy=-dy*0.005*60;lx=e.touches[0].clientX;ly=e.touches[0].clientY;}else if(e.touches.length===2){var dx=e.touches[0].clientX-e.touches[1].clientX;var dy=e.touches[0].clientY-e.touches[1].clientY;var dist=Math.sqrt(dx*dx+dy*dy);camDist*=ld/Math.max(dist,1);camDist=Math.max(0.3,Math.min(50,camDist));ld=dist;}},{passive:false});
canvas.addEventListener("touchend",function(e){e.preventDefault();if(e.touches.length===0)touching=false;},{passive:false});
canvas.addEventListener("mousedown",function(e){touching=true;vx=0;vy=0;lx=e.clientX;ly=e.clientY;});
canvas.addEventListener("mousemove",function(e){if(!touching)return;var dx=e.clientX-lx;var dy=e.clientY-ly;camTheta-=dx*0.005;camPhi+=dy*0.005;vx=dx*0.005*60;vy=-dy*0.005*60;lx=e.clientX;ly=e.clientY;});
canvas.addEventListener("mouseup",function(){touching=false;});
canvas.addEventListener("mouseleave",function(){touching=false;});
canvas.addEventListener("wheel",function(e){e.preventDefault();camDist*=(e.deltaY>0)?1.08:0.93;camDist=Math.max(0.3,Math.min(50,camDist));},{passive:false});
window.addEventListener("keydown",function(e){if(e.key==="r"){target=[cx,cy,cz];camTheta=0;camPhi=Math.PI/4;camDist=origCamDist;vx=0;vy=0;}});

function loadSplat(){
  if(SPLAT_B64){
    msg.textContent="Decoding base64...";
    var binary=atob(SPLAT_B64);
    var buffer=new ArrayBuffer(binary.length);
    var v=new Uint8Array(buffer);
    for(var i=0;i<binary.length;i++)v[i]=binary.charCodeAt(i);
    loadSplatData(buffer);
  }else if(SPLAT_URL){
    msg.textContent="Fetching "+SPLAT_URL+"...";
    var xhr=new XMLHttpRequest();
    xhr.open("GET",SPLAT_URL,true);
    xhr.responseType="arraybuffer";
    xhr.onload=function(){
      if(xhr.status===200||xhr.status===0){
        loadSplatData(xhr.response);
      }else{
        msg.textContent="HTTP "+xhr.status+". Try: python -m http.server";
      }
    };
    xhr.onerror=function(){msg.textContent="Load failed. Open via http:// not file://";};
    xhr.send();
    return;
  }else{
    msg.textContent="No data configured";
    return;
  }
  document.getElementById("info").textContent=N.toLocaleString()+" Gaussians - {{FILE_INFO}}";
  requestAnimationFrame(render);
}

loadSplat();
</script></body></html>'''
