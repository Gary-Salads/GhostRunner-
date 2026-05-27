#Project Ghost runner pdf for given batting average 
import gc
import math 
import matplotlib.pyplot as plt
# will look into using pybaseball, as of now I couldn't get it to work
#from pybaseball import playerid_lookup
#from pybaseball import batting_stats

def ghost_runner_pdf(batting_average, OPS):
    # Calculate the probability of a ghost runner scoring based on the batting average and OPS
    # This is a simplified model and may not reflect real-world baseball dynamics
    probability = batting_average * OPS  
    return probability
# Padres Batting lineup for 5/10 against the cardinals 
batting_line_up=["Shoei Ohtani", "Mookie Betts", "Freddie Freeman", "Kyle Tucker",
                  "Will Smith", "Max Muncy", "Andy Pages","Teoscar Hernandez", "HyeSeong Kim"]
season = 2026
# stats in order of batting lineup, want to use pybaseball to get these
ba = [0.268,0.175, 0.257,0.251,0.250,0.263,0.296,0.266,0.273]
OPS = [0.860,0.646,0.757,0.748,0.683,0.887,0.855,0.750,0.696]
y_values = []
for i in range(len(batting_line_up)):
    y_values.append(ghost_runner_pdf(ba[i], OPS[i]))

#clean up the plots
plt.close('all')
gc.collect()

# Plotting the PDF
plt.figure(figsize=(10, 6))
plt.bar(batting_line_up, y_values, color='blue')
plt.xlabel('Players')   
plt.ylabel('Probability of Ghost Runner Scoring')
plt.title('5/19 Dodgers Batting Lineup Ghost Runner PDF')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


