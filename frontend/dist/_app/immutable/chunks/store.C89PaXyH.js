import{d as v,g as a,h as _,u as g,i as m,j as l,k as i,l as h,m as k,n as w,o as x,v as b,w as y}from"./runtime.CW_FmKor.js";import{s as j}from"./index.DvAu-9B3.js";function A(e=!1){const s=_,n=s.l.u;if(!n)return;let t=()=>h(s.s);if(e){let u=0,r={};const d=k(()=>{let c=!1;const f=s.s;for(const o in f)f[o]!==r[o]&&(r[o]=f[o],c=!0);return c&&u++,u});t=()=>i(d)}n.b.length&&v(()=>{p(s,t),l(n.b)}),a(()=>{const u=g(()=>n.m.map(m));return()=>{for(const r of u)typeof r=="function"&&r()}}),n.a.length&&a(()=>{p(s,t),l(n.a)})}function p(e,s){if(e.l.s)for(const n of e.l.s)i(n);s()}function B(e,s,n){const t=n[s]??(n[s]={store:null,source:x(void 0),unsubscribe:b});if(t.store!==e)if(t.unsubscribe(),t.store=e??null,e==null)t.source.v=void 0,t.unsubscribe=b;else{var u=!0;t.unsubscribe=j(e,r=>{u?t.source.v=r:y(t.source,r)}),u=!1}return i(t.source)}function C(){const e={};return w(()=>{for(var s in e)e[s].unsubscribe()}),e}function D(e,s,n){return e.set(n),s}export{B as a,D as b,A as i,C as s};
