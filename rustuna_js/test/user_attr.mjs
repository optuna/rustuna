import assert from "node:assert";
import test from "node:test";

import * as rustuna from "../pkg/node/rustuna.js";

test("set_user_attr", () => {
	const study = rustuna.create_study("test");
	study.optimize((trial) => {
		const x = trial.suggest_float("x", -10.0, 10.0);
		trial.set_user_attr("foo", "bar");
		const value = (x - 5) ** 2;
		return value;
	}, 30);
	const best_trial = study.best_trial;
	assert.equal(best_trial.user_attrs.length, 1);
	assert.equal(best_trial.user_attrs[0].key, "foo");
	assert.equal(best_trial.user_attrs[0].value, "bar");
});
