Please notice the following:

- Thought Type variable reflects the degree to which participants were on-task. In other words, 7 = no mind wandering, 1 = mind wandering. If, for the sake of conceptual clarity, you wish to reverse this measure so it reflects mind wandering (with 7 = mind wandering, 1 = no mind wandering) you can use the following R code:


eyetracking_by_event_wide$Mindwandering <- 8-eyetracking_by_event_wide$Thought_Type

- All duration-based eye-tracking measures are measured in microseconds. If you wish to convert this to milliseconds (which is more common), simply divide the measure by 1,000.