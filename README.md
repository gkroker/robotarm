# robotarm
A two-joint robotarm, who catches balls with random flight-trajectories.

In data_generator.py, the the points of the flight-trajectories is randomly generated. This points are prepareted with a measurement 
uncertainty. After this, the points are exported to the file data.csv. In main.py, the catch-points of the ball in the configuration space 
of the robotarm is calculated. Linear regression is used to reconstructed a model of the flight-trajectories for the noisy coordinates. This 
model is used to predict the catchpoints as the the first point of the discretized flight-trajectory inside the configuration space of the 
robotarm.
