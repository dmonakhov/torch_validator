# torch_validator TODO

## Features

- [ ] Portable cache investigation - cache keys may include host-specific info
- [ ] Multi-host validation orchestration

## Code Quality

- [ ] Fix barrier() device warning - add device_id to init_process_group
- [ ] Fix NumPy warning - add as optional dependency or suppress
- [ ] Fix module import order warning in env_fingerprint

## Documentation

- [ ] Usage docs for local determinism test
- [ ] Usage docs for --portable flag
