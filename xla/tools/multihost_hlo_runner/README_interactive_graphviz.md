### Extract HLO subgraph (Optional)

You can use the XLA interactive_graphviz tool to extract a smaller portion of the model if desired.
Can be useful with the after optimization to share a smaller repro.

This example do the extract on an optimmized graph. This is more complex and request some flags to be able to run it.

```
bazel run //xla/tools:interactive_graphviz -- --hlo_text=/tmp/dump_multi_process/module_0023.pjit__wrapped_step_fn.sm_7.0_gpu_after_optimizations.txt
command: extract all-reduce-done.2 2
/tmp/tmp_file_tensorflow_1_TIy7uI.all-reduce-done.2-extracted.hlo

# Notes, this command line use *only* --run_xla_backend_only=true.
# Without it, it crashes. If you add --xla_disable_all_hlo_passes=true, it will also crash.
bazel run -c opt --config=cuda --dynamic_mode=off //xla/tools/multihost_hlo_runner:hlo_runner_main -- --run_xla_backend_only=true /tmp/tmp_file_tensorflow_1_TIy7uI.all-reduce-done.2-extracted.hlo
```

Then use that new hlo file.