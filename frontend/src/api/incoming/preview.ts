import { writable } from 'svelte/store';

export type VectorField = Array<{ x: number; y: number; z: number }>;
export type ScalarField = Array<Array<number>>;

export interface Preview {
	quantity: string;
	unit: string;
	component: string;
	layer: number;
	type: string;
	vectorFieldValues: VectorField;
	vectorFieldPositions: VectorField;
	scalarField: ScalarField;
	min: number;
	max: number;
	refresh: boolean;
	nComp: number;

	maxPoints: number;
	dataPointsCount: number;
	xPossibleSizes: number[]
	yPossibleSizes: number[]
	xChosenSize: number
	yChosenSize: number
	dynQuantities: { [key: string]: string[] };
	startX: number,
	startY: number,
	startZ: number,
	symmetricX: boolean,
	symmetricY: boolean,
	dynQuantitiesCat: string[]

}

export const previewState = writable<Preview>({
	quantity: '',
	unit: '',
	component: '',
	layer: 0,
	maxPoints: 0,
	type: '',
	vectorFieldValues: [],
	vectorFieldPositions: [],
	scalarField: [],
	min: 0,
	max: 0,
	refresh: false,
	nComp: 0,
	dataPointsCount: 0,
	xPossibleSizes: [],
	yPossibleSizes: [],
	xChosenSize: 0,
	yChosenSize: 0,
	dynQuantities: {},
	startX: 0,
	startY: 0,
	startZ: 0,
	symmetricX: false,
	symmetricY: false,
	dynQuantitiesCat: []
});
