# Custom Layout Simplification Notes

This folder stores custom RoboCasa kitchen layout YAML files used for lightweight
frontend exports.

## Layout 011 Notes

For `layout011_minimal.yaml`, the following parts are usually safe to remove or
disable when the goal is frontend scene display rather than RoboCasa task
simulation.

### Generally safe to disable

- Most or all `counter_accessories`
  - Examples: `toaster`, `coffee_machine`, `paper_towel`
  - Already-disabled examples in the source layout include `blender`,
    `electric_kettle`, `toaster_oven`, `dish_rack`, and `stand_mixer`.
  - These are usually leaf fixtures: they depend on counters or reference
    regions, but other major fixtures usually do not depend on them.

### Walls usually safe to remove

- `wall_front`
- `wall_front_backing`

These enclosing front walls are not needed for many frontend views, especially
when the scene is viewed from outside or above.

### Be careful with referenced fixtures

Do not remove a fixture if another fixture still references it through fields
such as:

- `align_to`
- `attach_to`
- `fixture`
- `ref`
- `interior_obj`
- `stack_fixtures`

If a fixture is removed, search for its name in the YAML and either remove the
dependent fixture too or retarget the dependency to another fixture.

### Recommended workflow

1. Disable or comment out leaf fixtures first.
2. Export and test after each small group of changes.
3. Remove backing walls/floors before removing major walls/floors.
4. Move slowly when changing counters, stacks, cabinets, sinks, or appliances,
   because these often serve as alignment anchors.
