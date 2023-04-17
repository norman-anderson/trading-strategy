import ManualStrategy
import experiment1, experiment2

import datetime as dt

if __name__ == "__main__":
    ### Manual Strategy Charts and Stats ###

    # In sample
    ManualStrategy.report(sd=dt.datetime(2008, 1, 1),
                          ed=dt.datetime(2009, 12, 31),
                          chart_name="In Sample")

    # Out of Sample
    ManualStrategy.report(sd=dt.datetime(2010, 1, 1),
                          ed=dt.datetime(2011, 12, 31),
                          chart_name="Out of Sample")


    ### Experiment 1 ###
    print("IN SAMPLE")
    experiment1.experiment1(dt.datetime(2008, 1, 1), dt.datetime(2009, 12, 31))

    print("OUT OF SAMPLE")
    experiment1.experiment1(dt.datetime(2010, 1, 1), dt.datetime(2011, 12, 31))

    #### Experiment 2 ####
    experiment2.experiment2()
