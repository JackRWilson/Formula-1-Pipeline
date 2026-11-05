select
    feature.race_id,
    race.season_year as year,
    race.round,
    circuit.circuit_id,
    circuit.circuit_name,
    driver.driver_id,
    driver.driver_name,
    constructor.constructor_id as team_id,
    constructor.constructor_name as team_name,
    result.finish_position as position,
    result.points,
    result.laps_completed,
    feature.feature_values
from formula_one.driver_race_feature_snapshot as feature
join formula_one.race as race
  on race.race_id = feature.race_id
join formula_one.driver as driver
  on driver.driver_id = feature.driver_id
left join formula_one.circuit as circuit
  on circuit.circuit_id = race.circuit_id
left join formula_one.constructor as constructor
  on constructor.constructor_id = feature.constructor_id
left join formula_one.race_result as result
  on result.race_id = feature.race_id
 and result.driver_id = feature.driver_id
where feature.prediction_stage = :prediction_stage
order by race.season_year, race.round, driver.driver_id
