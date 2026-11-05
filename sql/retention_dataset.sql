with constructor_counts as (
    select
        race.season_year,
        result.driver_id,
        result.constructor_id,
        count(*) as races_with_constructor
    from formula_one.race_result as result
    join formula_one.race as race
      on race.race_id = result.race_id
    where result.constructor_id is not null
    group by race.season_year, result.driver_id, result.constructor_id
),
ranked_constructors as (
    select
        season_year,
        driver_id,
        constructor_id,
        row_number() over (
            partition by season_year, driver_id
            order by races_with_constructor desc, constructor_id
        ) as team_rank
    from constructor_counts
),
season_metrics as (
    select
        race.season_year,
        result.driver_id,
        count(*) as races_participated,
        avg(result.starting_position::numeric) as avg_start_pos,
        stddev_samp(result.starting_position::numeric) as std_start_pos,
        min(result.starting_position) as best_start_pos,
        max(result.starting_position) as worst_start_pos,
        avg(result.finish_position::numeric) as avg_finish_pos,
        stddev_samp(result.finish_position::numeric) as std_finish_pos,
        min(result.finish_position) as best_finish_pos,
        max(result.finish_position) as worst_finish_pos,
        sum(result.points) as total_points,
        avg(result.points) as avg_points_per_race,
        sum(result.laps_completed) as total_laps_completed,
        count(*) filter (where result.finish_position = 1) as total_wins,
        count(*) filter (where result.finish_position between 1 and 3)
            as total_podium_finishes,
        count(*) filter (where result.points > 0) as total_points_finishes
    from formula_one.race_result as result
    join formula_one.race as race
      on race.race_id = result.race_id
    group by race.season_year, result.driver_id
),
driver_seasons as (
    select
        metrics.*,
        driver.driver_name,
        driver.continent,
        constructor.constructor_id as team_id,
        constructor.constructor_name as team_name,
        season.is_complete,
        row_number() over (
            partition by metrics.season_year
            order by metrics.total_points desc nulls last, metrics.driver_id
        ) as season_rank
    from season_metrics as metrics
    join ranked_constructors as ranked
      on ranked.season_year = metrics.season_year
     and ranked.driver_id = metrics.driver_id
     and ranked.team_rank = 1
    join formula_one.constructor as constructor
      on constructor.constructor_id = ranked.constructor_id
    join formula_one.driver as driver
      on driver.driver_id = metrics.driver_id
    join formula_one.season as season
      on season.season_year = metrics.season_year
),
retention_labels as (
    select
        current_season.*,
        case
            when next_season.is_complete
                then coalesce(next_driver_season.team_id = current_season.team_id, false)
            else null
        end as driver_retained
    from driver_seasons as current_season
    left join formula_one.season as next_season
      on next_season.season_year = current_season.season_year + 1
    left join driver_seasons as next_driver_season
      on next_driver_season.season_year = current_season.season_year + 1
     and next_driver_season.driver_id = current_season.driver_id
)
select
    season_year as year,
    driver_id,
    driver_name,
    continent,
    team_id,
    team_name,
    season_rank,
    avg_start_pos,
    std_start_pos,
    best_start_pos,
    worst_start_pos,
    races_participated,
    avg_finish_pos,
    std_finish_pos,
    best_finish_pos,
    worst_finish_pos,
    total_points,
    avg_points_per_race,
    total_laps_completed,
    total_wins,
    total_podium_finishes,
    total_points_finishes,
    driver_retained
from retention_labels
order by year, season_rank
