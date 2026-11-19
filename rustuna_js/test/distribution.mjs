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
	assert.equal(best_trial.params.length, 3);

	const x_param = best_trial.params.find((param) => param.name === "x");
	assert.equal(x_param.name, "x");
	assert.equal(
		x_param.internal_value >= -10 && x_param.internal_value <= 10,
		true,
	);

	const z_param = best_trial.params.find((param) => param.name === "z");
	assert.equal(z_param.name, "z");
	assert.equal(
		z_param.external_value === "foo" || z_param.external_value === "bar",
		true,
	);
});
