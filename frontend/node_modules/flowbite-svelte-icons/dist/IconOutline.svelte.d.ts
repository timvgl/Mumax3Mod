import type { Component } from 'svelte';
import type { SVGAttributes } from 'svelte/elements';
/**
 * [Go to docs](https://flowbite-svelte-icons.codewithshin.com/)
 * ## Props
 * @prop Icon
 * @prop size
 * @prop role
 * @prop color = 'currentColor'
 * @prop ariaLabel
 * @prop class: classname
 * @prop ...restProps
 */
declare const IconOutline: Component<SVGAttributes<SVGElement> & {
    Icon: Component;
    size?: "xs" | "sm" | "md" | "lg" | "xl";
    role?: string;
    color?: string;
    ariaLabel?: string;
    class?: string;
}, {}, "">;
export default IconOutline;
