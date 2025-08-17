from aisoccer.brains.BehindAndTowards import BehindAndTowards
from aisoccer.brains.DefendersAndAttackers import DefendersAndAttackers
from aisoccer.brains.RandomWalk import RandomWalk
from aisoccer.brains.SimpleBrain import SimpleBrain  # Add SimpleBrain
from aisoccer.brains.AdaptiveChaser import AdaptiveChaser  # Corrected import path
from aisoccer.brains.StrategicPlanner import StrategicPlanner  # Import StrategicPlanner
from aisoccer.tournament import *

brains = []
for i in range(4):  # Limit to 4 instances of each brain
    brains.append(BehindAndTowards("BAT-" + str(i)))
    brains.append(DefendersAndAttackers("DAA-" + str(i)))
    brains.append(RandomWalk("RAND-" + str(i)))
    brains.append(SimpleBrain("SIMPLE-" + str(i)))  # Add SimpleBrain instances
    brains.append(AdaptiveChaser("ADAPTIVE-" + str(i)))  # Add AdaptiveChaser instances
    brains.append(StrategicPlanner("STRATEGIC-" + str(i)))  # Add StrategicPlanner instances

tourney = Tournament(brains, game_length=5000, rounds=9)  # Double the game length
tourney.start()

def group_scores_by_brain_type(scores):
    grouped_scores = {}
    for score in scores:
        brain_type = score["name"].split('-')[0]  # Extract prefix
        if brain_type not in grouped_scores:
            grouped_scores[brain_type] = {'P': 0, 'W': 0, 'L': 0, 'GF': 0, 'GA': 0, 'GD': 0, 'POINTS': 0}
        grouped_scores[brain_type]['P'] += score['played']
        grouped_scores[brain_type]['W'] += score['wins']
        grouped_scores[brain_type]['L'] += score['losses']
        grouped_scores[brain_type]['GF'] += score['goals_for']
        grouped_scores[brain_type]['GA'] += score['goals_against']
        grouped_scores[brain_type]['GD'] += score['goal_diff']
        grouped_scores[brain_type]['POINTS'] += score['points']
    return grouped_scores

# Display grouped scores by brain type
scores = tourney.get_scores()
grouped_scores = group_scores_by_brain_type(scores)

# Sort grouped scores by points in descending order
sorted_grouped_scores = sorted(grouped_scores.items(), key=lambda x: x[1]['POINTS'], reverse=True)

print("\nGROUPED SCORES BY BRAIN TYPE:")
print(f"{'BRAIN TYPE':<10} | {'P':>5} | {'W':>5} | {'L':>5} | {'GF':>5} | {'GA':>5} | {'GD':>5} | {'POINTS':>7}")
for brain_type, score in sorted_grouped_scores:
    print(f"{brain_type:<10} | {score['P']:>5} | {score['W']:>5} | {score['L']:>5} | {score['GF']:>5} | {score['GA']:>5} | {score['GD']:>5} | {score['POINTS']:>7}")
