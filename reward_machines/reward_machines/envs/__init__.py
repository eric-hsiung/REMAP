from gym.envs.registration import register

# ----------------------------------------- Half-Cheetah

register(
    id='Half-Cheetah-RM1-v0',
    entry_point='envs.mujoco_rm.half_cheetah_environment:MyHalfCheetahEnvRM1',
    max_episode_steps=1000,
)
register(
    id='Half-Cheetah-RM2-v0',
    entry_point='envs.mujoco_rm.half_cheetah_environment:MyHalfCheetahEnvRM2',
    max_episode_steps=1000,
)



# ----------------------------------------- WATER
for i in range(11):
    w_id = 'Water-M%d-v0'%i
    w_en = 'envs.water.water_environment:WaterRMEnvM%d'%i
    register(
        id=w_id,
        entry_point=w_en,
        max_episode_steps=600
    )

for i in range(11):
    w_id = 'Water-single-M%d-v0'%i
    w_en = 'envs.water.water_environment:WaterRM10EnvM%d'%i
    register(
        id=w_id,
        entry_point=w_en,
        max_episode_steps=600
    )

# ----------------------------------------- OFFICE
register(
    id='Office-v0',
    entry_point='envs.grids.grid_environment:OfficeRMEnv',
    max_episode_steps=1000
)

register(
    id='Office-single-v0',
    entry_point='envs.grids.grid_environment:OfficeRM3Env',
    max_episode_steps=1000
)
register(
    id='Office-single-Test-T1-0-v0',
    entry_point='envs.grids.grid_environment:OfficeTestT1_0RMEnv',
    max_episode_steps=1000
)

register(
    id='Office-single-Reference-T1-v0',
    entry_point='envs.grids.grid_environment:OfficeRefT1RMEnv',
    max_episode_steps=1000
)
register(
    id='Office-single-Reference-T2-v0',
    entry_point='envs.grids.grid_environment:OfficeRefT2RMEnv',
    max_episode_steps=1000
)
register(
    id='Office-single-Reference-T3-v0',
    entry_point='envs.grids.grid_environment:OfficeRefT3RMEnv',
    max_episode_steps=1000
)
register(
    id='Office-single-Reference-T4-v0',
    entry_point='envs.grids.grid_environment:OfficeRefT4RMEnv',
    max_episode_steps=1000
)

register(
    id='Office-single-T0-v0',
    entry_point='envs.grids.grid_environment:OfficeLStarT0RMEnv',
    max_episode_steps=1000
)

register(
    id='Office-single-T1-v0',
    entry_point='envs.grids.grid_environment:OfficeLStarT1RMEnv',
    max_episode_steps=1000
)

register(
    id='Office-single-T2-v0',
    entry_point='envs.grids.grid_environment:OfficeLStarT2RMEnv',
    max_episode_steps=1000
)

register(
    id='Office-single-T3-v0',
    entry_point='envs.grids.grid_environment:OfficeLStarT3RMEnv',
    max_episode_steps=1000
)

register(
    id='Office-single-T4-v0',
    entry_point='envs.grids.grid_environment:OfficeLStarT4RMEnv',
    max_episode_steps=1000
)
# ----------------------------------------- CRAFT

## Craft Reference Environments -- specify the map and task in kwargs
register(
    id='Craft-single-Ref-v0',
    entry_point='envs.grids.grid_environment:CraftRefRMEnv',
    max_episode_steps=1000
)
## Craft Test  Environments -- specify the map and task and sample in kwargs
register(
    id='Craft-single-LStar-v0',
    entry_point='envs.grids.grid_environment:CraftLearnedRMEnv',
    max_episode_steps=1000
)
register(
    id='Craft-single-learned-LStar-v0',
    entry_point='envs.grids.grid_environment:CraftLearnedSingleRMEnv',
    max_episode_steps=1000
)

for i in range(11):
    w_id = 'Craft-M%d-v0'%i
    w_en = 'envs.grids.grid_environment:CraftRMEnvM%d'%i
    register(
        id=w_id,
        entry_point=w_en,
        max_episode_steps=1000
    )

for i in range(11):
    w_id = 'Craft-single-M%d-v0'%i
    w_en = 'envs.grids.grid_environment:CraftRM10EnvM%d'%i
    register(
        id=w_id,
        entry_point=w_en,
        max_episode_steps=1000
    )

for map_id in range(3):
    for task_id in range(1,11):
        w_id = f"Craft-single-M{map_id}-T{task_id}-v0"
        w_en = f"envs.grids.grid_environment:CraftLStarRMEnvM{map_id}T{task_id}"
        register(
            id=w_id,
            entry_point=w_en,
            max_episode_steps=1000
        )

for map_id in range(3):
    for task_id in range(1,11):
        w_id = f"Craft-single-Reference-M{map_id}-T{task_id}-v0"
        w_en = f"envs.grids.grid_environment:CraftRefRMEnvM{map_id}T{task_id}"
        register(
            id=w_id,
            entry_point=w_en,
            max_episode_steps=1000
        )
