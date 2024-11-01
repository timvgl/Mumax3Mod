import { SvelteComponentTyped } from "svelte";
import type { ActivityType } from '../types';
declare const __propDef: {
    props: {
        activities: ActivityType[];
        liClass?: string;
        spanClass?: string;
        imgClass?: string;
        outerDivClass?: string;
        innerDivClass?: string;
        timeClass?: string;
        titleClass?: string;
        textClass?: string;
    };
    events: {
        [evt: string]: CustomEvent<any>;
    };
    slots: {};
};
export type ActivityItemProps = typeof __propDef.props;
export type ActivityItemEvents = typeof __propDef.events;
export type ActivityItemSlots = typeof __propDef.slots;
/**
 * [Go to docs](https://flowbite-svelte.com/)
 * ## Props
 * @prop export let activities: $$Props['activities'] = [];
 * @prop export let liClass: $$Props['liClass'] = 'mb-10 ms-6';
 * @prop export let spanClass: $$Props['spanClass'] = 'flex absolute -start-3 justify-center items-center w-6 h-6 bg-blue-200 rounded-full ring-8 ring-white dark:ring-gray-900 dark:bg-blue-900';
 * @prop export let imgClass: $$Props['imgClass'] = 'rounded-full shadow-lg';
 * @prop export let outerDivClass: $$Props['outerDivClass'] = 'p-4 bg-white rounded-lg border border-gray-200 shadow-sm dark:bg-gray-700 dark:border-gray-600';
 * @prop export let innerDivClass: $$Props['innerDivClass'] = 'justify-between items-center mb-3 sm:flex';
 * @prop export let timeClass: $$Props['timeClass'] = 'mb-1 text-xs font-normal text-gray-400 sm:order-last sm:mb-0';
 * @prop export let titleClass: $$Props['titleClass'] = 'text-sm font-normal text-gray-500 lex dark:text-gray-300';
 * @prop export let textClass: $$Props['textClass'] = 'p-3 text-xs italic font-normal text-gray-500 bg-gray-50 rounded-lg border border-gray-200 dark:bg-gray-600 dark:border-gray-500 dark:text-gray-300';
 */
export default class ActivityItem extends SvelteComponentTyped<ActivityItemProps, ActivityItemEvents, ActivityItemSlots> {
}
export {};
