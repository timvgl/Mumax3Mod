import { SvelteComponentTyped } from "svelte";
import type { HTMLAttributes } from 'svelte/elements';
declare const __propDef: {
    props: HTMLAttributes<HTMLElement> & {
        steps: string[];
        currentStep: number;
        size: string;
        color: "primary" | "secondary" | "gray" | "red" | "yellow" | "green" | "indigo" | "purple" | "pink" | "blue" | "custom";
        glow: boolean;
        hideLabel: boolean;
        completedCustom: string;
        currentCustom: string;
    };
    events: {
        [evt: string]: CustomEvent<any>;
    };
    slots: {};
};
export type StepIndicatorProps = typeof __propDef.props;
export type StepIndicatorEvents = typeof __propDef.events;
export type StepIndicatorSlots = typeof __propDef.slots;
/**
 * [Go to docs](https://flowbite-svelte.com/)
 * ## Props
 * @prop export let steps: $$Props['steps'] = ['Step 1', 'Step 2', 'Step 3', 'Step 4', 'Step 5'];
 * @prop export let currentStep: $$Props['currentStep'] = 1;
 * @prop export let size: $$Props['size'] = 'h-2.5';
 * @prop export let color: $$Props['color'] = 'primary';
 * @prop export let glow: $$Props['glow'] = false;
 * @prop export let hideLabel: $$Props['hideLabel'] = false;
 * @prop export let completedCustom: $$Props['completedCustom'] = '';
 * @prop export let currentCustom: $$Props['currentCustom'] = '';
 */
export default class StepIndicator extends SvelteComponentTyped<StepIndicatorProps, StepIndicatorEvents, StepIndicatorSlots> {
}
export {};
