import{S as x,x as ee,y as re,z as w,A as ne,B as P,U as p,C as O,m,D as K,E as te,F as se,G as J,H as ie,I as M,J as Y,K as q,L as C,M as ae,N as fe,O as ue,P as le,Q as ce,R as U,T as oe,V as de,W as _e,X as Q,Y as ve,u as j,Z as $,_ as A,$ as W,a0 as pe,a1 as be,a2 as G,a3 as he,a4 as X,a5 as me,a6 as we,v as H,a7 as Pe,a8 as ye,a9 as xe,j as D,aa as k,i as ge}from"./utils.CzYqvc2b.js";import{d as Re}from"./disclose-version.DDWHpLfT.js";function E(e,r=null,i){if(typeof e!="object"||e===null||x in e)return e;const a=se(e);if(a!==ee&&a!==re)return e;var t=new Map,c=J(e),v=w(0);c&&t.set("length",w(e.length));var d;return new Proxy(e,{defineProperty(u,n,s){(!("value"in s)||s.configurable===!1||s.enumerable===!1||s.writable===!1)&&ne();var f=t.get(n);return f===void 0?(f=w(s.value),t.set(n,f)):P(f,E(s.value,d)),!0},deleteProperty(u,n){var s=t.get(n);if(s===void 0)n in u&&t.set(n,w(p));else{if(c&&typeof n=="string"){var f=t.get("length"),l=Number(n);Number.isInteger(l)&&l<f.v&&P(f,l)}P(s,p),Z(v)}return!0},get(u,n,s){var b;if(n===x)return e;var f=t.get(n),l=n in u;if(f===void 0&&(!l||(b=O(u,n))!=null&&b.writable)&&(f=w(E(l?u[n]:p,d)),t.set(n,f)),f!==void 0){var o=m(f);return o===p?void 0:o}return Reflect.get(u,n,s)},getOwnPropertyDescriptor(u,n){var s=Reflect.getOwnPropertyDescriptor(u,n);if(s&&"value"in s){var f=t.get(n);f&&(s.value=m(f))}else if(s===void 0){var l=t.get(n),o=l==null?void 0:l.v;if(l!==void 0&&o!==p)return{enumerable:!0,configurable:!0,value:o,writable:!0}}return s},has(u,n){var o;if(n===x)return!0;var s=t.get(n),f=s!==void 0&&s.v!==p||Reflect.has(u,n);if(s!==void 0||K!==null&&(!f||(o=O(u,n))!=null&&o.writable)){s===void 0&&(s=w(f?E(u[n],d):p),t.set(n,s));var l=m(s);if(l===p)return!1}return f},set(u,n,s,f){var g;var l=t.get(n),o=n in u;if(c&&n==="length")for(var b=s;b<l.v;b+=1){var y=t.get(b+"");y!==void 0?P(y,p):b in u&&(y=w(p),t.set(b+"",y))}l===void 0?(!o||(g=O(u,n))!=null&&g.writable)&&(l=w(void 0),P(l,E(s,d)),t.set(n,l)):(o=l.v!==p,P(l,E(s,d)));var h=Reflect.getOwnPropertyDescriptor(u,n);if(h!=null&&h.set&&h.set.call(f,s),!o){if(c&&typeof n=="string"){var I=t.get("length"),S=Number(n);Number.isInteger(S)&&S>=I.v&&P(I,S+1)}Z(v)}return!0},ownKeys(u){m(v);var n=Reflect.ownKeys(u).filter(l=>{var o=t.get(l);return o===void 0||o.v!==p});for(var[s,f]of t)f.v!==p&&!(s in u)&&n.push(s);return n},setPrototypeOf(){te()}})}function Z(e,r=1){P(e,e.v+r)}function z(e){return e!==null&&typeof e=="object"&&x in e?e[x]:e}function Le(e,r){return Object.is(z(e),z(r))}function Ce(e,r,i,a=null,t=!1){C&&ae();var c=e,v=null,d=null,u=null,n=t?fe:0;ie(()=>{if(u===(u=!!r()))return;let s=!1;if(C){const f=c.data===ue;u===f&&(c=le(),ce(c),U(!1),s=!0)}u?(v?M(v):v=Y(()=>i(c)),d&&q(d,()=>{d=null})):(d?M(d):a&&(d=Y(()=>a(c))),v&&q(v,()=>{v=null})),s&&U(!0)},n),C&&(c=oe)}const Ee={get(e,r){if(!e.exclude.includes(r))return e.props[r]},set(e,r){return!1},getOwnPropertyDescriptor(e,r){if(!e.exclude.includes(r)&&r in e.props)return{enumerable:!0,configurable:!0,value:e.props[r]}},has(e,r){return e.exclude.includes(r)?!1:r in e.props},ownKeys(e){return Reflect.ownKeys(e.props).filter(r=>!e.exclude.includes(r))}};function Ke(e,r,i){return new Proxy({props:e,exclude:r},Ee)}const Oe={get(e,r){if(!e.exclude.includes(r))return m(e.version),r in e.special?e.special[r]():e.props[r]},set(e,r,i){return r in e.special||(e.special[r]=Se({get[r](){return e.props[r]}},r,Q)),e.special[r](i),$(e.version),!0},getOwnPropertyDescriptor(e,r){if(!e.exclude.includes(r)&&r in e.props)return{enumerable:!0,configurable:!0,value:e.props[r]}},deleteProperty(e,r){return e.exclude.includes(r)||(e.exclude.push(r),$(e.version)),!0},has(e,r){return e.exclude.includes(r)?!1:r in e.props},ownKeys(e){return Reflect.ownKeys(e.props).filter(r=>!e.exclude.includes(r))}};function je(e,r){return new Proxy({props:e,exclude:r,special:{},version:w(0)},Oe)}const Ie={get(e,r){let i=e.props.length;for(;i--;){let a=e.props[i];if(A(a)&&(a=a()),typeof a=="object"&&a!==null&&r in a)return a[r]}},set(e,r,i){let a=e.props.length;for(;a--;){let t=e.props[a];A(t)&&(t=t());const c=O(t,r);if(c&&c.set)return c.set(i),!0}return!1},getOwnPropertyDescriptor(e,r){let i=e.props.length;for(;i--;){let a=e.props[i];if(A(a)&&(a=a()),typeof a=="object"&&a!==null&&r in a){const t=O(a,r);return t&&!t.configurable&&(t.configurable=!0),t}}},has(e,r){if(r===x||r===W)return!1;for(let i of e.props)if(A(i)&&(i=i()),i!=null&&r in i)return!0;return!1},ownKeys(e){const r=[];for(let i of e.props){A(i)&&(i=i());for(const a in i)r.includes(a)||r.push(a)}return r}};function Fe(...e){return new Proxy({props:e},Ie)}function V(e){for(var r=K,i=K;r!==null&&!(r.f&(pe|be));)r=r.parent;try{return G(r),e()}finally{G(i)}}function Se(e,r,i,a){var B;var t=(i&he)!==0,c=!X||(i&me)!==0,v=(i&we)!==0,d=(i&ye)!==0,u=!1,n;v?[n,u]=Re(()=>e[r]):n=e[r];var s=x in e||W in e,f=((B=O(e,r))==null?void 0:B.set)??(s&&v&&r in e?_=>e[r]=_:void 0),l=a,o=!0,b=!1,y=()=>(b=!0,o&&(o=!1,d?l=j(a):l=a),l);n===void 0&&a!==void 0&&(f&&c&&de(),n=y(),f&&f(n));var h;if(c)h=()=>{var _=e[r];return _===void 0?y():(o=!0,b=!1,_)};else{var I=V(()=>(t?H:Pe)(()=>e[r]));I.f|=_e,h=()=>{var _=m(I);return _!==void 0&&(l=void 0),_===void 0?l:_}}if(!(i&Q))return h;if(f){var S=e.$$legacy;return function(_,R){return arguments.length>0?((!c||!R||S||u)&&f(R?h():_),_):h()}}var g=!1,F=!1,N=xe(n),T=V(()=>H(()=>{var _=h(),R=m(N);return g?(g=!1,F=!0,R):(F=!1,N.v=_)}));return t||(T.equals=ve),function(_,R){if(arguments.length>0){const L=R?m(T):c&&v?E(_):_;return T.equals(L)||(g=!0,P(N,L),b&&l!==void 0&&(l=L),j(()=>m(T))),_}return m(T)}}function Be(e){D===null&&k(),X&&D.l!==null?Ae(D).m.push(e):ge(()=>{const r=j(e);if(typeof r=="function")return r})}function Te(e,r,{bubbles:i=!1,cancelable:a=!1}={}){return new CustomEvent(e,{detail:r,bubbles:i,cancelable:a})}function Me(){const e=D;return e===null&&k(),(r,i,a)=>{var c;const t=(c=e.s.$$events)==null?void 0:c[r];if(t){const v=J(t)?t.slice():[t],d=Te(r,i,a);for(const u of v)u.call(e.x,d);return!d.defaultPrevented}return!0}}function Ae(e){var r=e.l;return r.u??(r.u={a:[],b:[],m:[]})}export{E as a,Le as b,Me as c,Ce as i,je as l,Be as o,Se as p,Ke as r,Fe as s};
