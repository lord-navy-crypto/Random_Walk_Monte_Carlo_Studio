from __future__ import annotations
import json

DEFAULT_PRESET={
  "schema_version":3,
  "random_seed":12345,
  "random_walk":{"dimension":2,"steps":1000,"walkers":10000,"step_model":"uniform","p1":0.5,"p2":1.5},
  "batch_scan":{"variable":"n_steps","values":[10,100,1000,10000]},
  "repeated_mc":{"samples_per_trial":10000,"trials":100}
}

def dumps_preset(data)->str: return json.dumps(data,indent=2,sort_keys=True)

def loads_preset(text:str):
    data=json.loads(text)
    if not isinstance(data,dict): raise ValueError("Preset must be a JSON object")
    if data.get("schema_version",3)!=3: raise ValueError("Unsupported preset schema_version")
    rw=data.get("random_walk",{})
    required=["dimension","steps","walkers","step_model","p1"]
    missing=[k for k in required if k not in rw]
    if missing: raise ValueError(f"random_walk missing fields: {missing}")
    return data
