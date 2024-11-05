import{n as F,q as G,ae as N,a1 as S,ab as U,G as Y,a6 as V,D as T,af as A,ag as h,ah as P,ai as z,L as m,T as f,Q as D,M as W,aj as M,ak as J,al as Q,am as R,R as L,an as x,ao as K,ap as X,aq as Z,ar as ee,as as te,J as re,p as ae,a as ne,h as oe}from"./runtime.CW_FmKor.js";const $=new Set,k=new Set;function ie(e,t,r,i){function n(a){if(i.capture||E.call(t,a),!a.cancelBubble){var o=V,c=T;N(null),S(null);try{return r.call(this,a)}finally{N(o),S(c)}}}return e.startsWith("pointer")||e.startsWith("touch")||e==="wheel"?G(()=>{t.addEventListener(e,n,i)}):t.addEventListener(e,n,i),n}function pe(e,t,r,i,n){var a={capture:i,passive:n},o=ie(e,t,r,a);(t===document.body||t===window||t===document)&&F(()=>{t.removeEventListener(e,o,a)})}function ve(e){for(var t=0;t<e.length;t++)$.add(e[t]);for(var r of k)r(e)}function E(e){var C;var t=this,r=t.ownerDocument,i=e.type,n=((C=e.composedPath)==null?void 0:C.call(e))||[],a=n[0]||e.target,o=0,c=e.__root;if(c){var l=n.indexOf(c);if(l!==-1&&(t===document||t===window)){e.__root=t;return}var d=n.indexOf(t);if(d===-1)return;l<=d&&(o=l)}if(a=n[o]||e.target,a!==t){U(e,"currentTarget",{configurable:!0,get(){return a||r}});var w=V,_=T;N(null),S(null);try{for(var s,u=[];a!==null;){var v=a.assignedSlot||a.parentNode||a.host||null;try{var g=a["__"+i];if(g!==void 0&&!a.disabled)if(Y(g)){var[H,...j]=g;H.apply(a,[e,...j])}else g.call(a,e)}catch(b){s?u.push(b):s=b}if(e.cancelBubble||v===t||v===null)break;a=v}if(s){for(let b of u)queueMicrotask(()=>{throw b});throw s}}finally{e.__root=t,delete e.currentTarget,N(w),S(_)}}}function q(e){var t=document.createElement("template");return t.innerHTML=e,t.content}function p(e,t){var r=T;r.nodes_start===null&&(r.nodes_start=e,r.nodes_end=t)}function he(e,t){var r=(t&P)!==0,i=(t&z)!==0,n,a=!e.startsWith("<!>");return()=>{if(m)return p(f,null),f;n===void 0&&(n=q(a?e:"<!>"+e),r||(n=h(n)));var o=i?document.importNode(n,!0):n.cloneNode(!0);if(r){var c=h(o),l=o.lastChild;p(c,l)}else p(o,o);return o}}function me(e,t,r="svg"){var i=!e.startsWith("<!>"),n=(t&P)!==0,a=`<${r}>${i?e:"<!>"+e}</${r}>`,o;return()=>{if(m)return p(f,null),f;if(!o){var c=q(a),l=h(c);if(n)for(o=document.createDocumentFragment();h(l);)o.appendChild(h(l));else o=h(l)}var d=o.cloneNode(!0);if(n){var w=h(d),_=d.lastChild;p(w,_)}else p(d,d);return d}}function ge(e=""){if(!m){var t=A(e+"");return p(t,t),t}var r=f;return r.nodeType!==3&&(r.before(r=A()),D(r)),p(r,r),r}function ye(){if(m)return p(f,null),f;var e=document.createDocumentFragment(),t=document.createComment(""),r=A();return e.append(t,r),p(t,r),e}function we(e,t){if(m){T.nodes_end=f,W();return}e!==null&&e.before(t)}function Ee(e){return e.endsWith("capture")&&e!=="gotpointercapture"&&e!=="lostpointercapture"}const se=["beforeinput","click","change","dblclick","contextmenu","focusin","focusout","input","keydown","keyup","mousedown","mousemove","mouseout","mouseover","mouseup","pointerdown","pointermove","pointerout","pointerover","pointerup","touchend","touchmove","touchstart"];function Te(e){return se.includes(e)}const ue={formnovalidate:"formNoValidate",ismap:"isMap",nomodule:"noModule",playsinline:"playsInline",readonly:"readOnly"};function be(e){return e=e.toLowerCase(),ue[e]??e}const le=["touchstart","touchmove"];function de(e){return le.includes(e)}let I=!0;function Le(e){I=e}function Ne(e,t){var r=t==null?"":typeof t=="object"?t+"":t;r!==(e.__t??(e.__t=e.nodeValue))&&(e.__t=r,e.nodeValue=r==null?"":r+"")}function fe(e,t){return B(e,t)}function Se(e,t){M(),t.intro=t.intro??!1;const r=t.target,i=m,n=f;try{for(var a=h(r);a&&(a.nodeType!==8||a.data!==J);)a=Q(a);if(!a)throw R;L(!0),D(a),W();const o=B(e,{...t,anchor:a});if(f===null||f.nodeType!==8||f.data!==x)throw K(),R;return L(!1),o}catch(o){if(o===R)return t.recover===!1&&X(),M(),Z(r),L(!1),fe(e,t);throw o}finally{L(i),D(n)}}const y=new Map;function B(e,{target:t,anchor:r,props:i={},events:n,context:a,intro:o=!0}){M();var c=new Set,l=_=>{for(var s=0;s<_.length;s++){var u=_[s];if(!c.has(u)){c.add(u);var v=de(u);t.addEventListener(u,E,{passive:v});var g=y.get(u);g===void 0?(document.addEventListener(u,E,{passive:v}),y.set(u,1)):y.set(u,g+1)}}};l(ee($)),k.add(l);var d=void 0,w=te(()=>{var _=r??t.appendChild(A());return re(()=>{if(a){ae({});var s=oe;s.c=a}n&&(i.$$events=n),m&&p(_,null),I=o,d=e(_,i)||{},I=!0,m&&(T.nodes_end=f),a&&ne()}),()=>{var v;for(var s of c){t.removeEventListener(s,E);var u=y.get(s);--u===0?(document.removeEventListener(s,E),y.delete(s)):y.set(s,u)}k.delete(l),O.delete(d),_!==r&&((v=_.parentNode)==null||v.removeChild(_))}});return O.set(d,w),d}let O=new WeakMap;function Ae(e){const t=O.get(e);t&&t()}const ce="5";typeof window<"u"&&(window.__svelte||(window.__svelte={v:new Set})).v.add(ce);export{we as a,ge as b,ye as c,p as d,pe as e,q as f,Le as g,Se as h,Ee as i,ie as j,ve as k,be as l,fe as m,me as n,Te as o,I as p,Ne as s,he as t,Ae as u};