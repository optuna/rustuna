import assert from "node:assert";
import test from "node:test";

import * as rustuna from "../pkg/node/rustuna.js";

test("optimize simple quadratic function", () => {
	const study = rustuna.create_study("test");
	study.optimize((trial) => {
		const x = trial.suggest_float("x", -10.0, 10.0);
		const y = trial.suggest_int("y", -10, 10);
		trial.suggest_categorical("z", ["foo", "bar"]);
		const value = (x - 5) ** 2 + (y + 2) ** 2;
		return value;
	}, 30);
	const best_trial = study.best_trial;
	assert(best_trial.number < 30);
});

test("sample categorical values", () => {
	const study = rustuna.create_study("test");
	study.optimize((trial) => {
		trial.suggest_categorical("z", ["foo", 1, true, false, null]);
		// console.log(z);
		// console.log(typeof z);
		return 1.0;
	}, 5);
	const best_trial = study.best_trial;
	const external_value = best_trial.params[0].external_value;
	assert(["foo", 1, true, false, null].includes(external_value));
});
