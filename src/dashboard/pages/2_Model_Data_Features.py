import streamlit as st

st.set_page_config(page_title="Model Data Features", page_icon="⚙️", layout="wide")

st.title("⚙️ Model Data Features")

st.write("Here is an overview of all the data features I either extracted or engineered from available Formula 1 race data. The features have been divided up into 3 distinct sections. First, raw features consists of statistics that were readily available in the race data downloaded from `fastf1`. Second, calculated features include simple counts, sums, averages, standard deviations, and percentages that were created during the preprocessing phase. Finally, the third section is interaction features. These are model inputs that are a combination of one or more calculated statistics and were developed in the feature engineering phase of the project.")

st.write("As the generations of models were built out, some of these features were added or removed from model training. The :blue-background[blue] highlighted features are ones that were introduced when training the v1 and v2 models. The :gray-background[gray] highlighted features are ones that have been removed prior to training the v3 models. The :green-background[green] highlighted features are ones that were only introduced when designing and developing the v3 generation.")

st.write("The features without any highlight are ones that were used for identifying records or as metadata; they were not used to train any model.")

st.subheader("Raw Features")
st.write("*Note: The race data from `fastf1` was precented by driver, so I had to consolidate records for each team. I didn't use many statistics straight from the race data which is why this list is small.")

st.markdown("""
- :blue-background[Year]: The F1 season year. Provides temporal context for changes in team performance across seasons.
- :blue-background[Round]: The current race round number in the season (e.g., Round 7 = seventh race). Useful for capturing progression.
- Location: The location of the current race round.
- TeamName: The official name of the constructor team.
- TeamId: The `fastf1` shorthand id for a team (in the event the team name changed mid-season).
- DriverId: The `fastf1` shorthand id for a particular driver.
""")

st.subheader("Calculated Features")

st.markdown("""
- :gray-background[RoundsCompleted]: The total number of races completed prior to the current round.
- :blue-background[RoundsRemaining]: The total number of races left in the season after the current round. Helps estimate how much a team can still gain.
- :blue-background[PointsEarnedThisRound]: The constructor's total points earned in the current race.
- :gray-background[MaxDriverPoints]: The maximum number of points one driver within a team earned in a particular round.
- :gray-background[MinDriverPoints]: The minimum number of points one driver within a team earned in a particular round.
- :blue-background[DNFsThisRound]: The number of did not finish (DNF) results for the team in the current round (only values of 0, 1, or 2).
- :blue-background[PointsLast3Rounds]: The rolling sum of points earned for a team over the last 3 races. Helps capture recent, short-term form.
- :blue-background[DNFsLast3Rounds]: The rolling sum of DNFs over the last 3 races. Highlights recent reliability issues.
- :blue-background[DNFRate]: The proportion of races in the season so far where the team recorded at least 1 DNF. Highlights long-term reliability issues.
- :blue-background[AvgGridPosition]: The team's average qualifying grid position for each round.
- :blue-background[AvgPosition]: The team's average finishing position for each round.
- :blue-background[AvgPointsPerRace]: The mean amount of points earned per race to date. Provides a standardized measure of performance.
- :blue-background[TotalPointFinishes]: The count of races where the team scored at least 1 point. Reflects consistency in reaching the top 10.
- :blue-background[TotalPodiums]: The cumulative count of podium finishes by a team in the current season.
- :blue-background[TotalPoints]: The cumulative sum of points earned by a team up to a particular round in the season.
- :blue-background[hadPenaltyThisYear]: A binary flag (0 or 1) which indicates whether the team has incurred a points deduction penalty during the season. *(For example: In 2020, Racing Point received a 15 point penalty for copying the brake ducts from the 2019 Mercedes car).*
""")

st.subheader("Interaction Features")

st.markdown("""
- :gray-background[AvgPointsLast5Rounds]: The rolling average of points earned by a team in the last 5 rounds.
- :gray-background[StdDevLast5Rounds]: The rolling standard deviation of points earned by a team in the last 5 rounds.
- :green-background[DriverPointsGap]: The absolute points difference between the two drivers in the team. Helps capture intra-team balance or imbalance.            
- :green-background[RelativePointsShare]: The ratio of one team's points to the total points earned by all the constructor teams in the season so far. Normalizes performance by grid competitiveness.
- :blue-background[PercentileRankAfterRound]: The percentile standing of the constructor's points after each round relative to the field.
""")

st.markdown(" - :green-background[Consistency]")
st.latex('''
    \\frac{1}{(1 + (\\text{StdDevLast5Rounds} / (\\text{AvgPointsLast5Rounds} + 1e^{-6})))}
''')
st.write("The standard deviation of points earned across the last 5 races in one season. A low standard deviation indicates driver consistency and car reliability.")

st.markdown(" - :green-background[FormRatio]")
st.latex('''
    \\frac{\\text{PointsLast3Rounds}}{\\text{AvgPointsPerRace} * 3}
''')
st.write("Values > 1 indicate an improving form whereas values < 1 suggest a decline.")

st.markdown(" - :gray-background[ProjectedGrowth]")
st.latex('''
    \\text{AvgPointsLast5Rounds} * \\text{RoundsRemaining}
''')

st.markdown(" - :green-background[ProjectedSeasonTotalPoints]")
st.latex('''
    \\text{TotalPoints} + \\text{ProjectedGrowth}
''')
st.write("This feature provides a forward-looking performance estimate *without* using future data.")

