## In terminal one, go to ETAP repo
python scripts/inference_cear_lcm.py --config config/exe/inference_online/feature_tracking_cear.yaml 

## In terminal two, go to ETAP repo and publish manually defined feature positions and timestamp range
python scripts/feature_position_lcm_publisher.py --manual 100 100 130 210 --start-us 1704749447959841 --end-us 1704749448959841

## Explaination
The first command will let ETAP wait for features to be published onto a topic. We want it to always wait for new data coming. That means after it finishes inferencing data, it will go back to waiting stage.

The second command will manually publish two feature positions (100, 100), (130, 210) with start timestamp 1704749447959841 and end timestamp 1704749448959841 in microseconds. Here I manually add 1 second to get end timestamp.

## Goal:

1. check if tracked points are accurate at new timestamp.
2. make sure the ETAP will go back to waiting stage after finish inference. You can check this by sending command in terminal two again: python scripts/feature_position_lcm_publisher.py --manual 105 100 135 210 --start-us 1704749448959841 --end-us 1704749449959841, and check if the predicted results are accurate.