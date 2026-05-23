#!/usr/bin/env python3
"""Create a minimal point-cloud HTML preview from a .splat file."""

import argparse
import base64
from pathlib import Path


TEMPLATE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title>
<style>
html,body,canvas{{margin:0;width:100%;height:100%;overflow:hidden;background:#11131f}}
#info{{position:fixed;left:12px;bottom:10px;color:#d8dbe8;font:12px system-ui,sans-serif}}
</style>
</head>
<body>
<canvas id="c"></canvas><div id="info"></div>
<script>
const b64 = "{b64}";
const bin = atob(b64);
const buf = new ArrayBuffer(bin.length);
const u8 = new Uint8Array(buf);
for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
const view = new DataView(buf);
const n = Math.floor(buf.byteLength / 32);
const pos = new Float32Array(n * 3);
const color = new Float32Array(n * 3);
let cx = 0, cy = 0, cz = 0;
for (let i = 0; i < n; i++) {{
  const off = i * 32;
  const x = view.getFloat32(off, true);
  const y = view.getFloat32(off + 4, true);
  const z = view.getFloat32(off + 8, true);
  pos[i * 3] = x; pos[i * 3 + 1] = y; pos[i * 3 + 2] = z;
  color[i * 3] = u8[off + 24] / 255;
  color[i * 3 + 1] = u8[off + 25] / 255;
  color[i * 3 + 2] = u8[off + 26] / 255;
  cx += x; cy += y; cz += z;
}}
cx /= n; cy /= n; cz /= n;
let maxR = 1;
for (let i = 0; i < n; i++) {{
  const dx = pos[i * 3] - cx, dy = pos[i * 3 + 1] - cy, dz = pos[i * 3 + 2] - cz;
  maxR = Math.max(maxR, Math.hypot(dx, dy, dz));
}}

const canvas = document.getElementById("c");
const gl = canvas.getContext("webgl");
const vs = gl.createShader(gl.VERTEX_SHADER);
gl.shaderSource(vs, `
attribute vec3 p; attribute vec3 c; uniform mat4 mvp; varying vec3 vc;
void main(){{ gl_Position = mvp * vec4(p, 1.0); gl_PointSize = 2.0; vc = c; }}
`);
gl.compileShader(vs);
const fs = gl.createShader(gl.FRAGMENT_SHADER);
gl.shaderSource(fs, `precision mediump float; varying vec3 vc; void main(){{ gl_FragColor = vec4(vc, 1.0); }}`);
gl.compileShader(fs);
const prog = gl.createProgram();
gl.attachShader(prog, vs); gl.attachShader(prog, fs); gl.linkProgram(prog); gl.useProgram(prog);
function attr(name, data) {{
  const loc = gl.getAttribLocation(prog, name);
  const b = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, b);
  gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
  gl.enableVertexAttribArray(loc); gl.vertexAttribPointer(loc, 3, gl.FLOAT, false, 0, 0);
}}
attr("p", pos); attr("c", color);
const mvpLoc = gl.getUniformLocation(prog, "mvp");
function mmul(a,b){{const o=new Float32Array(16);for(let r=0;r<4;r++)for(let c=0;c<4;c++)o[c*4+r]=a[0*4+r]*b[c*4+0]+a[1*4+r]*b[c*4+1]+a[2*4+r]*b[c*4+2]+a[3*4+r]*b[c*4+3];return o;}}
function look(eye,tgt){{let zx=eye[0]-tgt[0],zy=eye[1]-tgt[1],zz=eye[2]-tgt[2];let zl=Math.hypot(zx,zy,zz);zx/=zl;zy/=zl;zz/=zl;let xx=-zy,xy=zx,xz=0;let xl=Math.hypot(xx,xy,xz)||1;xx/=xl;xy/=xl;let yx=zy*xz-zz*xy,yy=zz*xx-zx*xz,yz=zx*xy-zy*xx;return new Float32Array([xx,yx,zx,0,xy,yy,zy,0,xz,yz,zz,0,-(xx*eye[0]+xy*eye[1]+xz*eye[2]),-(yx*eye[0]+yy*eye[1]+yz*eye[2]),-(zx*eye[0]+zy*eye[1]+zz*eye[2]),1]);}}
function proj(fov,aspect,near,far){{const f=1/Math.tan(fov/2),nf=1/(near-far);return new Float32Array([f/aspect,0,0,0,0,f,0,0,0,0,(far+near)*nf,-1,0,0,2*far*near*nf,0]);}}
let theta = 0;
function frame() {{
  canvas.width = innerWidth * devicePixelRatio; canvas.height = innerHeight * devicePixelRatio;
  gl.viewport(0,0,canvas.width,canvas.height); gl.clearColor(0.07,0.08,0.12,1); gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);
  theta += 0.006;
  const d = Math.max(4, maxR * 2.8);
  const eye = [cx + Math.cos(theta) * d, cy + Math.sin(theta) * d, cz + d * 0.45];
  gl.uniformMatrix4fv(mvpLoc, false, mmul(proj(Math.PI/3, canvas.width/canvas.height, 0.01, 1000), look(eye, [cx,cy,cz])));
  gl.drawArrays(gl.POINTS, 0, n);
  requestAnimationFrame(frame);
}}
document.getElementById("info").textContent = `${{n.toLocaleString()}} points, radius ${{maxR.toFixed(2)}}`;
frame();
</script>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("splat")
    parser.add_argument("html")
    args = parser.parse_args()
    data = base64.b64encode(Path(args.splat).read_bytes()).decode("ascii")
    Path(args.html).write_text(TEMPLATE.format(title=Path(args.splat).stem, b64=data), encoding="utf-8")
    print(args.html)


if __name__ == "__main__":
    main()
