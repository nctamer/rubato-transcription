"use strict";
const paper = document.querySelector(".paper-card");
const light = document.querySelector(".spotlight");
const logo = document.querySelector(".title-logo");
const fine = matchMedia("(hover: hover) and (pointer: fine)");
const reduced = matchMedia("(prefers-reduced-motion: reduce)");
let frame = null, lastTime = null, initialized = false, inside = false;
let x = 0, y = 0, vx = 0, vy = 0, targetX = 0, targetY = 0;
// Animate only the composited transform. Keep velocity when the target changes.
light.style.left = "0px";
light.style.top = "0px";
light.style.transition = "none";
function paint(){
  light.style.transform = `translate3d(${x}px, ${y}px, 0) translate(-50%, -50%)`;
}
function tick(time){
  const elapsed = lastTime === null ? 1 / 60 : Math.min((time-lastTime)/1000, 1/30);
  lastTime = time;
  const steps = Math.max(1,Math.ceil(elapsed / (1/120)));
  const dt = elapsed / steps;
  // A soft pursuit; a slightly quicker, well-damped return to the logo.
  const stiffness = inside ? 100 : 185;
  const damping = inside ? 17 : 25;
  for(let i=0;i<steps;i++){
    vx += ((targetX-x)*stiffness-vx*damping)*dt;
    vy += ((targetY-y)*stiffness-vy*damping)*dt;
    x += vx*dt; y += vy*dt;
  }
  if(Math.hypot(targetX-x,targetY-y)<.12 && Math.hypot(vx,vy)<.7){
    x=targetX; y=targetY; vx=0; vy=0; paint();
    frame=null; lastTime=null; return;
  }
  paint(); frame=requestAnimationFrame(tick);
}
function aim(px,py,snap=false){
  targetX=px; targetY=py;
  if(snap || reduced.matches){
    if(frame!==null)cancelAnimationFrame(frame);
    frame=null; lastTime=null; x=px; y=py; vx=0; vy=0; paint(); return;
  }
  if(frame===null){lastTime=null;frame=requestAnimationFrame(tick);}
}
function home(snap=false){
  inside=false;
  const cell=paper.getBoundingClientRect(), anchor=logo.getBoundingClientRect();
  aim(anchor.left-cell.left+anchor.width/2,anchor.top-cell.top+anchor.height/2,snap || !initialized);
  initialized=true;
}
function follow(event){
  if(!fine.matches || reduced.matches || event.pointerType==="touch")return;
  inside=true;
  const cell=paper.getBoundingClientRect();
  aim(event.clientX-cell.left,event.clientY-cell.top);
}
paper.addEventListener("pointerenter",follow);
paper.addEventListener("pointermove",follow);
paper.addEventListener("pointerleave",()=>home());
paper.addEventListener("pointercancel",()=>home());
fine.addEventListener("change",()=>home(true));
reduced.addEventListener("change",()=>home(true));
window.addEventListener("resize",()=>home(true));
window.addEventListener("load",()=>{if(!inside)home(true);});
if(document.fonts)document.fonts.ready.then(()=>{if(!inside)home(true);});
home(true);
document.getElementById("copy-citation").addEventListener("click",async()=>{
 const status=document.getElementById("copy-status");
 try{await navigator.clipboard.writeText(document.getElementById("bibtex").textContent);status.textContent="Citation copied.";}
 catch{const range=document.createRange();range.selectNodeContents(document.getElementById("bibtex"));const selection=getSelection();selection.removeAllRanges();selection.addRange(range);status.textContent="Citation selected. Copy it or download the .bib file.";}
});

document.getElementById("open-citation").addEventListener("click", () => {
  document.getElementById("citation").open = true;
});
