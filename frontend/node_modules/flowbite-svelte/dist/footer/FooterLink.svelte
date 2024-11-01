<script>import { twMerge } from "tailwind-merge";
export let liClass = "me-4 last:me-0 md:me-6";
export let aClass = "hover:underline";
export let href = "";
export let target = void 0;
let liCls = twMerge(liClass, $$props.classLi);
let aCls = twMerge(aClass, $$props.classA);
</script>

<li class={liCls}>
  <a {...$$restProps} {href} class={aCls} {target}>
    <slot />
  </a>
</li>

<!--
@component
[Go to docs](https://flowbite-svelte.com/)
## Props
@prop export let liClass: $$Props['liClass'] = 'me-4 last:me-0 md:me-6';
@prop export let aClass: $$Props['aClass'] = 'hover:underline';
@prop export let href: $$Props['href'] = '';
@prop export let target: $$Props['target'] = undefined;
-->
