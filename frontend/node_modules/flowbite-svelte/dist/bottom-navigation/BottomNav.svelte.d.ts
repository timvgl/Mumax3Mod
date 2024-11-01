import { SvelteComponentTyped } from "svelte";
export type BottomNavType = {
    activeClass: string;
    btnClass?: string;
    spanClass?: string;
};
import type { HTMLAttributes } from 'svelte/elements';
declare const __propDef: {
    props: HTMLAttributes<HTMLDivElement> & {
        activeUrl: string;
        position?: "static" | "fixed" | "absolute" | "relative" | "sticky";
        navType?: "default" | "border" | "application" | "pagination" | "group" | "card" | "meeting" | "video";
        outerClass?: string;
        innerClass?: string;
        activeClass?: string;
    };
    events: {
        [evt: string]: CustomEvent<any>;
    };
    slots: {
        header: {};
        default: {};
    };
};
export type BottomNavProps = typeof __propDef.props;
export type BottomNavEvents = typeof __propDef.events;
export type BottomNavSlots = typeof __propDef.slots;
/**
 * [Go to docs](https://flowbite-svelte.com/)
 * ## Props
 * @prop export let activeUrl: $$Props['activeUrl'] = '';
 * @prop export let position: $$Props['position'] = 'fixed';
 * @prop export let navType: NonNullable<$$Props['navType']> = 'default';
 * @prop export let outerClass: $$Props['outerClass'] = 'w-full z-50 border-gray-200 dark:bg-gray-700 dark:border-gray-600';
 * @prop export let innerClass: $$Props['innerClass'] = 'grid h-full max-w-lg mx-auto';
 * @prop export let activeClass: $$Props['activeClass'] = 'text-primary-700 dark:text-primary-700 hover:text-primary-900 dark:hover:text-primary-900';
 */
export default class BottomNav extends SvelteComponentTyped<BottomNavProps, BottomNavEvents, BottomNavSlots> {
}
export {};
