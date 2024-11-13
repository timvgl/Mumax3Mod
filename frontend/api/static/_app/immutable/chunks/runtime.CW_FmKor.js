var Dn=Array.isArray,Cn=Array.from,Nn=Object.defineProperty,ft=Object.getOwnPropertyDescriptor,Jt=Object.getOwnPropertyDescriptors,bn=Object.prototype,qn=Array.prototype,Qt=Object.getPrototypeOf;function Pn(t){return typeof t=="function"}const Fn=()=>{};function Ln(t){return t()}function ht(t){for(var n=0;n<t.length;n++)t[n]()}const T=2,dt=4,Y=8,nt=16,w=32,z=64,x=128,V=256,h=512,g=1024,b=2048,N=4096,j=8192,Wt=16384,Et=32768,Mn=65536,Xt=1<<18,yt=1<<19,_t=Symbol("$state"),Hn=Symbol("");function wt(t){return t===this.v}function tn(t,n){return t!=t?n==n:t!==n||t!==null&&typeof t=="object"||typeof t=="function"}function Tt(t){return!tn(t,this.v)}function nn(t){throw new Error("effect_in_teardown")}function rn(){throw new Error("effect_in_unowned_derived")}function en(t){throw new Error("effect_orphan")}function sn(){throw new Error("effect_update_depth_exceeded")}function Yn(){throw new Error("hydration_failed")}function jn(t){throw new Error("props_invalid_value")}function Bn(){throw new Error("state_descriptors_fixed")}function Un(){throw new Error("state_prototype_fixed")}function an(){throw new Error("state_unsafe_local_read")}function un(){throw new Error("state_unsafe_mutation")}function rt(t){return{f:0,v:t,reactions:null,equals:wt,version:0}}function Vn(t){return mt(rt(t))}function ln(t,n=!1){var e;const r=rt(t);return n||(r.equals=Tt),i!==null&&i.l!==null&&((e=i.l).s??(e.s=[])).push(r),r}function Gn(t,n=!1){return mt(ln(t,n))}function mt(t){return o!==null&&o.f&T&&(y===null?An([t]):y.push(t)),t}function Kn(t,n){return et(t,Zt(()=>it(t))),n}function et(t,n){return o!==null&&ot()&&o.f&(T|nt)&&(y===null||!y.includes(t))&&un(),on(t,n)}function on(t,n){return t.equals(n)||(t.v=n,t.version=Bt(),At(t,g),ot()&&l!==null&&l.f&h&&!(l.f&w)&&(v!==null&&v.includes(t)?(E(l,g),J(l)):I===null?In([t]):I.push(t))),n}function At(t,n){var r=t.reactions;if(r!==null)for(var e=ot(),s=r.length,a=0;a<s;a++){var u=r[a],f=u.f;f&g||!e&&u===l||(E(u,n),f&(h|x)&&(f&T?At(u,b):J(u)))}}const $n=1,Zn=2,zn=4,Jn=8,Qn=16,Wn=1,Xn=2,tr=4,nr=8,rr=16,er=4,sr=1,ar=2,fn="[",_n="[!",cn="]",It={},ur=Symbol(),lr="http://www.w3.org/2000/svg";function gt(t){console.warn("hydration_mismatch")}let k=!1;function or(t){k=t}let _;function L(t){if(t===null)throw gt(),It;return _=t}function ir(){return L(R(_))}function fr(t){if(k){if(R(_)!==null)throw gt(),It;_=t}}function _r(t=1){if(k){for(var n=t,r=_;n--;)r=R(r);_=r}}function cr(){for(var t=0,n=_;;){if(n.nodeType===8){var r=n.data;if(r===cn){if(t===0)return n;t-=1}else(r===fn||r===_n)&&(t+=1)}var e=R(n);n.remove(),n=e}}var ct,St,kt;function vr(){if(ct===void 0){ct=window;var t=Element.prototype,n=Node.prototype;St=ft(n,"firstChild").get,kt=ft(n,"nextSibling").get,t.__click=void 0,t.__className="",t.__attributes=null,t.__e=void 0,Text.prototype.__t=void 0}}function st(t=""){return document.createTextNode(t)}function Q(t){return St.call(t)}function R(t){return kt.call(t)}function pr(t){if(!k)return Q(t);var n=Q(_);return n===null&&(n=_.appendChild(st())),L(n),n}function hr(t,n){if(!k){var r=Q(t);return r instanceof Comment&&r.data===""?R(r):r}if(n&&(_==null?void 0:_.nodeType)!==3){var e=st();return _==null||_.before(e),L(e),e}return _}function dr(t,n=1,r=!1){let e=k?_:t;for(;n--;)e=R(e);if(!k)return e;var s=e.nodeType;if(r&&s!==3){var a=st();return e==null||e.before(a),L(a),a}return L(e),e}function Er(t){t.textContent=""}function vn(t){var n=T|g;l===null?n|=x:l.f|=yt;const r={children:null,deps:null,equals:wt,f:n,fn:t,reactions:null,v:null,version:0,parent:l};if(o!==null&&o.f&T){var e=o;(e.children??(e.children=[])).push(r)}return r}function yr(t){const n=vn(t);return n.equals=Tt,n}function xt(t){var n=t.children;if(n!==null){t.children=null;for(var r=0;r<n.length;r+=1){var e=n[r];e.f&T?at(e):P(e)}}}function Rt(t){var n,r=l;Z(t.parent);try{xt(t),n=Ut(t)}finally{Z(r)}return n}function Ot(t){var n=Rt(t),r=(O||t.f&x)&&t.deps!==null?b:h;E(t,r),t.equals(n)||(t.v=n,t.version=Bt())}function at(t){xt(t),H(t,0),E(t,j),t.v=t.children=t.deps=t.reactions=null}function Dt(t){l===null&&o===null&&en(),o!==null&&o.f&x&&rn(),lt&&nn()}function pn(t,n){var r=n.last;r===null?n.last=n.first=t:(r.next=t,t.prev=r,n.last=t)}function q(t,n,r,e=!0){var s=(t&z)!==0,a=l,u={ctx:i,deps:null,deriveds:null,nodes_start:null,nodes_end:null,f:t|g,first:null,fn:n,last:null,next:null,parent:s?null:a,prev:null,teardown:null,transitions:null,version:0};if(r){var f=D;try{vt(!0),B(u),u.f|=Wt}catch(p){throw P(u),p}finally{vt(f)}}else n!==null&&J(u);var m=r&&u.deps===null&&u.first===null&&u.nodes_start===null&&u.teardown===null&&(u.f&yt)===0;if(!m&&!s&&e&&(a!==null&&pn(u,a),o!==null&&o.f&T)){var c=o;(c.children??(c.children=[])).push(u)}return u}function wr(t){const n=q(Y,null,!1);return E(n,h),n.teardown=t,n}function Tr(t){Dt();var n=l!==null&&(l.f&w)!==0&&i!==null&&!i.m;if(n){var r=i;(r.e??(r.e=[])).push({fn:t,effect:l,reaction:o})}else{var e=Ct(t);return e}}function mr(t){return Dt(),ut(t)}function Ar(t){const n=q(z,t,!0);return()=>{P(n)}}function Ct(t){return q(dt,t,!1)}function Ir(t,n){var r=i,e={effect:null,ran:!1};r.l.r1.push(e),e.effect=ut(()=>{t(),!e.ran&&(e.ran=!0,et(r.l.r2,!0),Zt(n))})}function gr(){var t=i;ut(()=>{if(it(t.l.r2)){for(var n of t.l.r1){var r=n.effect;r.f&h&&E(r,b),F(r)&&B(r),n.ran=!1}t.l.r2.v=!1}})}function ut(t){return q(Y,t,!0)}function Sr(t){return hn(t)}function hn(t,n=0){return q(Y|nt|n,t,!0)}function kr(t,n=!0){return q(Y|w,t,!0,n)}function Nt(t){var n=t.teardown;if(n!==null){const r=lt,e=o;pt(!0),$(null);try{n.call(null)}finally{pt(r),$(e)}}}function bt(t){var n=t.deriveds;if(n!==null){t.deriveds=null;for(var r=0;r<n.length;r+=1)at(n[r])}}function qt(t,n=!1){var r=t.first;for(t.first=t.last=null;r!==null;){var e=r.next;P(r,n),r=e}}function dn(t){for(var n=t.first;n!==null;){var r=n.next;n.f&w||P(n),n=r}}function P(t,n=!0){var r=!1;if((n||t.f&Xt)&&t.nodes_start!==null){for(var e=t.nodes_start,s=t.nodes_end;e!==null;){var a=e===s?null:R(e);e.remove(),e=a}r=!0}bt(t),qt(t,n&&!r),H(t,0),E(t,j);var u=t.transitions;if(u!==null)for(const m of u)m.stop();Nt(t);var f=t.parent;f!==null&&f.first!==null&&Pt(t),t.next=t.prev=t.teardown=t.ctx=t.deps=t.parent=t.fn=t.nodes_start=t.nodes_end=null}function Pt(t){var n=t.parent,r=t.prev,e=t.next;r!==null&&(r.next=e),e!==null&&(e.prev=r),n!==null&&(n.first===t&&(n.first=e),n.last===t&&(n.last=r))}function xr(t,n){var r=[];Ft(t,r,!0),En(r,()=>{P(t),n&&n()})}function En(t,n){var r=t.length;if(r>0){var e=()=>--r||n();for(var s of t)s.out(e)}else n()}function Ft(t,n,r){if(!(t.f&N)){if(t.f^=N,t.transitions!==null)for(const u of t.transitions)(u.is_global||r)&&n.push(u);for(var e=t.first;e!==null;){var s=e.next,a=(e.f&Et)!==0||(e.f&w)!==0;Ft(e,n,a?r:!1),e=s}}}function Rr(t){Lt(t,!0)}function Lt(t,n){if(t.f&N){t.f^=N,F(t)&&B(t);for(var r=t.first;r!==null;){var e=r.next,s=(r.f&Et)!==0||(r.f&w)!==0;Lt(r,s?n:!1),r=e}if(t.transitions!==null)for(const a of t.transitions)(a.is_global||n)&&a.in()}}const yn=typeof requestIdleCallback>"u"?t=>setTimeout(t,1):requestIdleCallback;let G=!1,K=!1,W=[],X=[];function Mt(){G=!1;const t=W.slice();W=[],ht(t)}function Ht(){K=!1;const t=X.slice();X=[],ht(t)}function Or(t){G||(G=!0,queueMicrotask(Mt)),W.push(t)}function Dr(t){K||(K=!0,yn(Ht)),X.push(t)}function wn(){G&&Mt(),K&&Ht()}function Cr(){throw new Error("invalid_default_snippet")}function Tn(t){throw new Error("lifecycle_outside_component")}const Yt=0,mn=1;let U=Yt,M=!1,D=!1,lt=!1;function vt(t){D=t}function pt(t){lt=t}let S=[],C=0;let o=null;function $(t){o=t}let l=null;function Z(t){l=t}let y=null;function An(t){y=t}let v=null,d=0,I=null;function In(t){I=t}let jt=0,O=!1,i=null;function Bt(){return++jt}function ot(){return i!==null&&i.l===null}function F(t){var u,f;var n=t.f;if(n&g)return!0;if(n&b){var r=t.deps,e=(n&x)!==0;if(r!==null){var s;if(n&V){for(s=0;s<r.length;s++)((u=r[s]).reactions??(u.reactions=[])).push(t);t.f^=V}for(s=0;s<r.length;s++){var a=r[s];if(F(a)&&Ot(a),e&&l!==null&&!O&&!((f=a==null?void 0:a.reactions)!=null&&f.includes(t))&&(a.reactions??(a.reactions=[])).push(t),a.version>t.version)return!0}}e||E(t,h)}return!1}function gn(t,n,r){throw t}function Ut(t){var A;var n=v,r=d,e=I,s=o,a=O,u=y,f=t.f;v=null,d=0,I=null,o=f&(w|z)?null:t,O=!D&&(f&x)!==0,y=null;try{var m=(0,t.fn)(),c=t.deps;if(v!==null){var p;if(H(t,d),c!==null&&d>0)for(c.length=d+v.length,p=0;p<v.length;p++)c[d+p]=v[p];else t.deps=c=v;if(!O)for(p=d;p<c.length;p++)((A=c[p]).reactions??(A.reactions=[])).push(t)}else c!==null&&d<c.length&&(H(t,d),c.length=d);return m}finally{v=n,d=r,I=e,o=s,O=a,y=u}}function Sn(t,n){let r=n.reactions;if(r!==null){var e=r.indexOf(t);if(e!==-1){var s=r.length-1;s===0?r=n.reactions=null:(r[e]=r[s],r.pop())}}r===null&&n.f&T&&(v===null||!v.includes(n))&&(E(n,b),n.f&(x|V)||(n.f^=V),H(n,0))}function H(t,n){var r=t.deps;if(r!==null)for(var e=n;e<r.length;e++)Sn(t,r[e])}function B(t){var n=t.f;if(!(n&j)){E(t,h);var r=l,e=i;l=t,i=t.ctx;try{bt(t),n&nt?dn(t):qt(t),Nt(t);var s=Ut(t);t.teardown=typeof s=="function"?s:null,t.version=jt}catch(a){gn(a)}finally{l=r,i=e}}}function Vt(){C>1e3&&(C=0,sn()),C++}function Gt(t){var n=t.length;if(n!==0){Vt();var r=D;D=!0;try{for(var e=0;e<n;e++){var s=t[e];s.f&h||(s.f^=h);var a=[];Kt(s,a),kn(a)}}finally{D=r}}}function kn(t){var n=t.length;if(n!==0)for(var r=0;r<n;r++){var e=t[r];!(e.f&(j|N))&&F(e)&&(B(e),e.deps===null&&e.first===null&&e.nodes_start===null&&(e.teardown===null?Pt(e):e.fn=null))}}function xn(){if(M=!1,C>1001)return;const t=S;S=[],Gt(t),M||(C=0)}function J(t){U===Yt&&(M||(M=!0,queueMicrotask(xn)));for(var n=t;n.parent!==null;){n=n.parent;var r=n.f;if(r&(z|w)){if(!(r&h))return;n.f^=h}}S.push(n)}function Kt(t,n){var r=t.first,e=[];t:for(;r!==null;){var s=r.f,a=(s&w)!==0,u=a&&(s&h)!==0;if(!u&&!(s&N))if(s&Y){a?r.f^=h:F(r)&&B(r);var f=r.first;if(f!==null){r=f;continue}}else s&dt&&e.push(r);var m=r.next;if(m===null){let A=r.parent;for(;A!==null;){if(t===A)break t;var c=A.next;if(c!==null){r=c;continue t}A=A.parent}}r=m}for(var p=0;p<e.length;p++)f=e[p],n.push(f),Kt(f,n)}function $t(t){var n=U,r=S;try{Vt();const s=[];U=mn,S=s,M=!1,Gt(r);var e=t==null?void 0:t();return wn(),(S.length>0||s.length>0)&&$t(),C=0,e}finally{U=n,S=r}}async function Nr(){await Promise.resolve(),$t()}function it(t){var f;var n=t.f,r=(n&T)!==0;if(r&&n&j){var e=Rt(t);return at(t),e}if(o!==null){y!==null&&y.includes(t)&&an();var s=o.deps;v===null&&s!==null&&s[d]===t?d++:v===null?v=[t]:v.push(t),I!==null&&l!==null&&l.f&h&&!(l.f&w)&&I.includes(t)&&(E(l,g),J(l))}else if(r&&t.deps===null){var a=t,u=a.parent;u!==null&&!((f=u.deriveds)!=null&&f.includes(a))&&(u.deriveds??(u.deriveds=[])).push(a)}return r&&(a=t,F(a)&&Ot(a)),t.v}function Zt(t){const n=o;try{return o=null,t()}finally{o=n}}const Rn=~(g|b|h);function E(t,n){t.f=t.f&Rn|n}function br(t){return zt().get(t)}function qr(t,n){return zt().set(t,n),n}function zt(t){return i===null&&Tn(),i.c??(i.c=new Map(On(i)||void 0))}function On(t){let n=t.p;for(;n!==null;){const r=n.c;if(r!==null)return r;n=n.p}return null}function Pr(t,n=1){var r=+it(t);return et(t,r+n),r}function Fr(t,n=!1,r){i={p:i,c:null,e:null,m:!1,s:t,x:null,l:null},n||(i.l={s:null,u:null,r1:[],r2:rt(!1)})}function Lr(t){const n=i;if(n!==null){const u=n.e;if(u!==null){var r=l,e=o;n.e=null;try{for(var s=0;s<u.length;s++){var a=u[s];Z(a.effect),$(a.reaction),Ct(a.fn)}}finally{Z(r),$(e)}}i=n.p,n.m=!0}return{}}function Mr(t){if(!(typeof t!="object"||!t||t instanceof EventTarget)){if(_t in t)tt(t);else if(!Array.isArray(t))for(let n in t){const r=t[n];typeof r=="object"&&r&&_t in r&&tt(r)}}}function tt(t,n=new Set){if(typeof t=="object"&&t!==null&&!(t instanceof EventTarget)&&!n.has(t)){n.add(t),t instanceof Date&&t.getTime();for(let e in t)try{tt(t[e],n)}catch{}const r=Qt(t);if(r!==Object.prototype&&r!==Array.prototype&&r!==Map.prototype&&r!==Set.prototype&&r!==Date.prototype){const e=Jt(r);for(let s in e){const a=e[s].get;if(a)try{a.call(t)}catch{}}}}}export{w as $,rt as A,Bn as B,ft as C,l as D,Un as E,Qt as F,Dn as G,hn as H,Rr as I,kr as J,xr as K,k as L,ir as M,Et as N,_n as O,cr as P,L as Q,or as R,_t as S,_ as T,ur as U,jn as V,Mn as W,tr as X,Tt as Y,Pr as Z,Pn as _,Lr as a,z as a0,Z as a1,Wn as a2,Xn as a3,nr as a4,yr as a5,o as a6,j as a7,rr as a8,Tn as a9,on as aA,Zn as aB,Qn as aC,Ft as aD,En as aE,P as aF,zn as aG,Jn as aH,lr as aI,ot as aJ,qr as aK,Kn as aL,Dr as aM,Hn as aN,Jt as aO,nt as aP,Wt as aQ,er as aR,Cr as aS,$t as aa,Nn as ab,Nr as ac,Vn as ad,$ as ae,st as af,Q as ag,sr as ah,ar as ai,vr as aj,fn as ak,R as al,It as am,cn as an,gt as ao,Yn as ap,Er as aq,Cn as ar,Ar as as,Ir as at,gr as au,Gn as av,br as aw,_r as ax,N as ay,$n as az,fr as b,pr as c,mr as d,Ct as e,hr as f,Tr as g,i as h,Ln as i,ht as j,it as k,Mr as l,vn as m,wr as n,ln as o,Fr as p,Or as q,ut as r,dr as s,Sr as t,Zt as u,Fn as v,et as w,tn as x,bn as y,qn as z};