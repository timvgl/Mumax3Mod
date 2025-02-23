import { previewState } from '$api/incoming/preview';
import { get, writable } from 'svelte/store';

export const quantities: { [category: string]: string[] } = {
	Common: ['m', 'torque', 'regions', 'Msat', 'Aex', 'alpha'],

	'Magnetic Fields': [
		'B_anis',
		'B_custom',
		'B_demag',
		'B_eff',
		'B_exch',
		'B_ext',
		'B_mel',
		'B_therm'
	],

	Energy: [
		'Edens_anis',
		'Edens_custom',
		'Edens_demag',
		'Edens_exch',
		'Edens_mel',
		'Edens_therm',
		'Edens_total',
		'Edens_Zeeman',
		'Edens_el',
		'Edens_kin'
	],

	'Force': [
		'F_melM',
		'F_el',
		'F_elsys',
		'rhod2udt2',
		'etadudt'
	],

	Anisotropy: ['anisC1', 'anisC2', 'anisU', 'Kc1', 'Kc2', 'Kc3', 'Ku1', 'Ku2'],

	DMI: ['Dbulk', 'Dind', 'DindCoupling'],

	External: [
		'ext_bubbledist',
		'ext_bubblepos',
		'ext_bubblespeed',
		'ext_corepos',
		'ext_dwpos',
		'ext_dwspeed',
		'ext_dwtilt',
		'ext_dwxpos',
		'ext_topologicalcharge',
		'ext_topologicalchargedensity',
		'ext_topologicalchargedensitylattice',
		'ext_topologicalchargelattice'
	],

	'Spin-transfer Torque': ['xi', 'STTorque'],

	Strain: ['exx', 'exy', 'exz', 'eyy', 'eyz', 'ezz'],

	Current: ['J', 'Pol'],

	Slonczewski: ['EpsilonPrime', 'FixedLayer', 'FreeLayerThickness', 'Lambda'],

	'Magneto-elastic-constants': ['B1', 'B2', 'C11', 'C12', 'C44', 'eta', 'rho'],

	'Magneto-elastic-dynamics': ['F_mel', 'u', 'du', 'normStrain', 'normStress', 'shearStrain', 'shearStress', 'force_density', 'poynting'],


	Miscellaneous: [
		'frozenspins',
		'NoDemagSpins',
		'MFM',
		'spinAngle',
		'LLtorque',
		'm_full',
		'Temp',
		'geom'
	]
};

export const dynQuantities = writable<{ [key: string]: string[] }>({});
export const dynQuantitiesCat = writable<string[]>([]);
export function updateDynQuantites() {
	const ps = get(previewState);
    dynQuantitiesCat.set(ps.dynQuantitiesCat || []);
	dynQuantities.set(ps.dynQuantities || {});
};

