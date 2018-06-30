import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
import pymc3 as pm 
import theano.tensor as tt
import theano



def run_games(a, b, samples=len(trace)):

    '''
    Get result of a match between a and b, 'sample' times. 
    sample should be an integer >=1. 
    a, b are intengers, correspondng to indices ofthe 'all_teams' array
    '''

    with team_abilities:
        home_team.set_value([a])
        away_team.set_value([b])

        ppc=pm.sample_ppc(trace, progressbar=False, samples=samples)

    return ppc

def match(a, b, samples=len(trace)):
    '''
    Simulate one match (or 'sample' matches, for sample an integer >1) between two teams a and b, and save the results nicely. a, b are integers corresponding to indices of the 'all_teams' array
    '''

    ppc=run_games(a, b, samples=samples)

    A=pd.DataFrame(data=np.array([ppc['home_scored'], ppc['away_scored']]).T, columns=['goals_for', 'goals_against'])

    B=pd.DataFrame(data=np.array([ppc['away_scored'], ppc['home_scored']]).T, columns=['goals_for', 'goals_against'])

    
    A_win=(A.goals_for>A.goals_against)
    draw=(A.goals_for==A.goals_against)
    A_loss=(A.goals_for<A.goals_against)

    A['points']=np.zeros_like(A_win, dtype=int)
    B['points']=np.zeros_like(A_win, dtype=int) 

    A.points[A_win]=3.0
    A.points[A_loss]=0.0
    A.points[draw]=1.0

    B.points[A_win]=0.0
    B.points[A_loss]=3.0
    B.points[draw]=3.0

    return A, B

def knockout_match(a, b, samples=1):

    '''
    Run a knockout game. If there's a draw, the teams go to a penalty shootout with 50% chance of winning each
    '''

    ppc=run_games(a, b, samples=samples)

    A=pd.DataFrame(data=np.array([ppc['home_scored'], ppc['away_scored']]).T, columns=['goals_for', 'goals_against'])
    B=pd.DataFrame(data=np.array([ppc['away_scored'], ppc['home_scored']]).T, columns=['goals_for', 'goals_against'])

    A_win=(A.goals_for>A.goals_against)
    A_loss=(A.goals_for<A.goals_against)

    normal_time_draw=(A.goals_for==A.goals_against)
    
    #Say that after a normal time draw, go to penalties. Chance of progresisng is 0.5
    penalties=np.random.randint(2, size=np.sum(normal_time_draw)).astype(bool)

    A_win[normal_time_draw]=penalties
    A_loss[normal_time_draw]=~penalties


    A['win']=A_win
    B['win']=A_loss

    return A, B, A.win.values[0], B.win.values[0]


def knockout_rounds(teams, results):
    '''
    Run a knockout round from start to finish
    '''

    #Knockout round is AvB, CvD, EvF, GvH, IvJ, KvL, MvN, OvP
    #                      AvC,     EvG      IvK       MvO
    #                            AvE            IvM
    #                                 AvI
    teams=np.array(teams)

    #Mask with which we'll knock teams out with
    mask=np.ones_like(teams, dtype=bool)

    #Names of the rounds
    rnds=['L16', 'QF', 'SF', 'F', 'W']
    for i, rnd in enumerate(rnds[:-1]):
        #Loop through each match up
        for pair in teams[mask].reshape(-1, 2):

            a=np.where(all_teams==(pair[0]))[0][0]
            b=np.where(all_teams==(pair[1]))[0][0]

            results.loc[pair[0]]=i+1
            results.loc[pair[1]]=i+1

            #Find the indices of each team in the list of teams...
            #...in order to update the mask
            team_1_index=np.where(teams==pair[0])
            team_2_index=np.where(teams==pair[1])

            #Play the match
            A, B, team_1_win, team_2_win=knockout_match(a, b)
            assert team_1_win + team_2_win ==1, 'Whoops! Two wins/two losses'
            
            #See which team progresses
            results.loc[pair[0]]+=team_1_win
            results.loc[pair[1]]+=team_2_win

            #Update the mask
            mask[team_1_index]=team_1_win
            mask[team_2_index]=team_2_win

            print("{} ({}) vs {} ({}): {}-{} ({}-{})".format(pair[0], a,  pair[1], b, A.goals_for.values, B.goals_for.values, team_1_win, team_2_win))


    return teams[mask], results

def world_cup(knockout_teams):

    '''
    Simulate a whole world cup- uncomment to also simulate a group stage too
    '''


    # group_a=["Russia", "Saudi Arabia", "Egypt", "Uruguay"]
    # group_b=["Portugal", "Spain", "Morocco", "Iran"]
    # group_c=["France", "Australia", "Peru", "Denmark"]
    # group_d=["Argentina", "Iceland", "Croatia", "Nigeria"]
    # group_e=["Brazil", "Switzerland", "Costa Rica", "Serbia"]
    # group_f=["Germany", "Mexico", "Sweden", "South Korea"]
    # group_g=["Belgium", "Panama", "Tunisia", "England"]
    # group_h=["Poland", "Senegal", "Colombia", "Japan"]

    # index_tuples=pd.MultiIndex.from_tuples([*list(zip(['A']*4, group_a)),
    #  *list(zip(['B']*4, group_b)),
    #  *list(zip(['C']*4, group_c)),
    #  *list(zip(['D']*4, group_d)),
    #  *list(zip(['E']*4, group_e)),
    #  *list(zip(['F']*4, group_f)),
    #  *list(zip(['G']*4, group_g)),
    #  *list(zip(['H']*4, group_h))])

    # all_results_groupstages=pd.Series(np.zeros(32), index=index_tuples)
    # all_results_groupstages=all_results_groupstages.swaplevel()
    

    # letters=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    # groups=dict(zip(letters, [group_a, group_b, group_c, group_d, group_e, group_f, group_g, group_h]))

    # #Simulate the matches
    # for L in letters:

    #     all_results_groupstages=group(L, groups, all_results_groupstages)
    
    # #Got all the results from the groupstage
    # #Switch indices and final group position order
    # gs=[*['A']*4, *['B']*4, *['C']*4, *['D']*4, *['E']*4, *['F']*4, *['G']*4, *['H']*4]
    # pos=all_results_groupstages.values
    # index=pd.MultiIndex.from_arrays((gs, pos)) 

    # countries_in_order=all_results_groupstages.index.levels[0][all_results_groupstages.index.labels[0]]

    # R=pd.Series(countries_in_order, index=index)

    all_results_knockouts=pd.Series(np.zeros(16), index=knockout_teams)

    # knockout_teams=[R[('C', 1)], R[('D', 2)], R[('A', 1)], R[('B', 2)],
    # R[('G', 1)], R[('H', 2)], R[('E', 1)], R[('F', 2)], R[('C', 2)], R[('D', 1)], R[('B', 1)], R[('A', 2)], R[('G', 2)], R[('H', 1)], R[('E', 2)], R[('F', 1)]]

    winner, final_results=knockout_rounds(knockout_teams, all_results_knockouts)

    return final_results, winner

def group(group_letter, groups, all_results, samples=1):
    '''
    Simulate a group stage with given teams
    '''

    names=groups[group_letter]
    #Matches are (ab, cd), (ac, bd), (ad, bc)
    a=np.where(all_teams==(names[0]))[0][0]
    b=np.where(all_teams==(names[1]))[0][0]
    c=np.where(all_teams==(names[2]))[0][0]
    d=np.where(all_teams==(names[3]))[0][0]

    A1, B1=match(a, b, samples=samples)
    C1, D1=match(c, d, samples=samples)
    A2, C2=match(a, c, samples=samples)
    B2, D2=match(b, d, samples=samples)
    A3, D3=match(a, d, samples=samples)
    B3, C3=match(b, c, samples=samples)

    A=A1+A2+A3
    B=B1+B2+B3
    C=C1+C2+C3
    D=D1+D2+D3

    for tab, tm in zip([A, B, C, D], names):
        tab['gd']=tab.goals_for-tab.goals_against
        tab['team']=tm

    #Concatenate each team's results into a table, with multi-indices 'points' and 'gd' on one level, and 'A', 'B', 'C', 'D' on the second level.
    table=pd.concat([A, B, C, D]).pivot(columns='team', values=['points', 'gd'])

    #Find the final position in the group, sorting first on points and then on goal difference
    #The last bit reverses the order, so it sorts descending
    places=np.lexsort((table['gd'].values, table['points'].values))[:, ::-1]
    places=places.squeeze()

    #The final standings are:
    standings=table['points'].T.index.values[places]

    results=np.apply_along_axis(lambda a: np.histogram(a, bins=4)[0], 0, places)
    probs=results/np.sum(results, axis=0)


    for i, name in enumerate(table['points'].T.index.values):
        #table[('standing', '{}'.format(name))]=places[:, i]+1
        all_results.loc[(group_letter, name)]=np.where(standings==name)[0][0]+1
        
    
    return all_results





def show_group_probabilities(probs, teams, ax=None):
    '''
    Show where teams finish if we simulate a group
    '''

    #Add P(qualifying) axis
    probs=np.column_stack((probs, np.sum(probs[:, :2], axis=1)))

    if ax is None:
        fig, ax=plt.subplots()

    ax.matshow(probs, cmap='plasma', alpha=0.8)

    for (i, j), z in np.ndenumerate(probs):
        ax.text(j, i, '{:0.1f}%'.format(z*100.0), ha='center', va='center')

    ax.set_yticklabels(np.insert(teams, 0, ''))
    xticks=[0, 1, 2, 3, 4, "P(Qualify)"]
    ax.set_xticklabels(xticks)

    ax.set_xlabel('Final Group Position')
    ax.xaxis.set_label_position('top') 

    return fig, ax


def make_violinplot(t, n, inds, cmap='plasma', context='seaborn-white', **kwargs):
    '''
    Make a violin plot. 
    t- numpy array of traces, shape [9000, 20]
    n- names of teams for the y labels
    inds- indices which sort the teams into top and bottom 20
    '''

    xlabel=kwargs.pop('xlabel', '')
    title=kwargs.pop('title', '')
    import matplotlib as mpl
    #Make a violin plot
    with plt.style.context((context)):
        fig, ax=plt.subplots(figsize=(10, 10))
        parts=ax.violinplot(t, positions=np.delete(np.arange(21), 10), vert=False, showextrema=True, showmedians=True)
        ax.axhline(10.0, linestyle='dashed', linewidth=2.0, c='k')
        #Details
        ax.set_yticks(np.arange(0, 21))
        #ax.set_xticks(np.arange(1, 11)/10.)
        ax.set_yticklabels(list(n), fontsize=20)
        ax.tick_params(axis='both', labelsize=20)
        ax.set_xlabel(xlabel, fontsize=20)
        ax.set_title(title, fontsize=25)

        #Colour the violins by number of games
        cm=plt.get_cmap(cmap)

        norm = mpl.colors.Normalize(vmin=n_matches.min(), vmax=n_matches.max())
        for pc, ind in zip(parts['bodies'], inds):

            c=cm(n_matches.loc[all_teams[ind]])
            pc.set_facecolor(c)
            pc.set_edgecolor('k')
            pc.set_alpha(1)

        sm = plt.cm.ScalarMappable(cmap=cm, norm=norm)
        sm._A = []
        cb = fig.colorbar(sm, ax=ax)
        cb.set_label(r'# of matches', fontsize=20)

        fig.tight_layout()
    return fig, ax

if __name__=='__main__':
    from tqdm import tqdm

    #Load the table
    df=pd.read_csv('results.csv')
    wc_df=pd.read_csv('wc_matches.csv')
    wc_teams=wc_df.team1.unique()

    #Get rid of teams with few games and make sure all world cup nations are represented
    home_team_games=df.groupby(['home_team']).size() 
    all_teams=set(home_team_games[home_team_games.sort_values()>100].index.values)
    all_teams.add('Serbia')
    all_teams.add('Iceland')
    all_teams.add('Croatia')
    all_teams.add('Panama')

    all_teams=np.array(list(all_teams))
    all_teams[np.where(all_teams=='Korea Republic')]='South Korea'


    #Get rid of some of the smaller teams
    all_teams=np.delete(all_teams, [np.where(all_teams=='Uganda'), np.where(all_teams=='Zambia'), np.where(all_teams=='Malaysia'), np.where(['Czechoslovakia'])])



    #Add some columns- currently not used in the model
    tmp1 = df[['date','home_team', 'home_score']]
    tmp2 = df[['date','away_team', 'away_score']]
    tmp1.columns =  ['Date','name', 'score']
    tmp2.columns =  ['Date','name', 'score']
    tmp = pd.concat([tmp1,tmp2], ignore_index=True)
    df['home_last_5_goals_for']=tmp.sort_values(by='Date').groupby("name")["score"].rolling(5).sum().shift(1).reset_index(0, drop=True)

    tmp = pd.concat([tmp2,tmp1], ignore_index=True)
    df['away_last_5_goals_for']=tmp.sort_values(by='Date').groupby("name")["score"].rolling(5).sum().shift(1).reset_index(0, drop=True)

    tmp1 = df[['date','home_team', 'away_score']]
    tmp2 = df[['date','away_team', 'home_score']]
    tmp1.columns =  ['Date','name', 'score']
    tmp2.columns =  ['Date','name', 'score']
    tmp = pd.concat([tmp1,tmp2], ignore_index=True)
    df['home_last_5_goals_against']=tmp.sort_values(by='Date').groupby("name")["score"].rolling(5).sum().shift(1).reset_index(0, drop=True)

    tmp = pd.concat([tmp2,tmp1], ignore_index=True)
    df['away_last_5_goals_against']=tmp.sort_values(by='Date').groupby("name")["score"].rolling(5).sum().shift(1).reset_index(0, drop=True)

    df['home_last_5_gd']=df['home_last_5_goals_for']-df['home_last_5_goals_against']
    df['away_last_5_gd']=df['away_last_5_goals_for']-df['away_last_5_goals_against']

    #Drop the first row of each group, since it's been contaminated by rolling the last row to the start.
    #Here we make it nan and then dropna 
    def mask_first(x):
        result = np.ones_like(x)
        result[0] = 0
        return result
    mask = ~df.groupby(['home_team'])['home_score'].transform(mask_first).astype(bool)
    df.loc[mask]=np.nan
    df=df.dropna()



    #Get theteams in each match
    teams = pd.DataFrame(all_teams, columns=['team'])
    teams['i'] = teams.index
    mask=(df.home_team.isin(all_teams))&(df.away_team.isin(all_teams))
    df=df[mask]


    #Merge
    df = pd.merge(df, teams, left_on='home_team', right_on='team', how='left')
    df = df.rename(columns = {'i': 'i_home'}).drop('team', 1)
    df = pd.merge(df, teams, left_on='away_team', right_on='team', how='left')
    df = df.rename(columns = {'i': 'i_away'}).drop('team', 1)

    #Only use matches since 2000
    from datetime import datetime
    cutoff=datetime(2000, 1, 1)
    df.date=pd.to_datetime(df['date'])
    df=df[df.date>cutoff]
    tmp.Date=pd.to_datetime(tmp['Date'])


    n_matches=tmp[tmp.Date>cutoff].name.value_counts()
    n_teams=len(np.unique(all_teams))


    #Theano shared variables, so we can update them later
    home_team = theano.shared(df.i_home.values)
    away_team = theano.shared(df.i_away.values)

    #Home Goals and Away goals for each match
    observed_home_goals = df.home_score.values
    observed_away_goals = df.away_score.values

    # #Goal Difference in last 5 games-not used at the moment
    # home_last_5_gd=df.home_last_5_gd.values
    # away_last_5_gd=df.away_last_5_gd.values

    # #zscore 
    # home_gd_z=theano.shared((home_last_5_gd-np.mean(home_last_5_gd))/np.std(home_last_5_gd))
    # away_gd_z=theano.shared((away_last_5_gd-np.mean(away_last_5_gd))/np.std(away_last_5_gd))

    #The pymc3 model
    with pm.Model() as team_abilities:

        #Priors
        intercept = pm.Flat('intercept')
        sd_att = pm.HalfNormal('sd_att', sd=10**2)
        sd_def = pm.HalfNormal('sd_def', sd=10**2)


        # team-specific model parameters
        atts_star = pm.Normal("atts_star", mu=0, sd=sd_att, shape=n_teams)
        defs_star = pm.Normal("defs_star", mu=0, sd=sd_def, shape=n_teams)

        atts = pm.Deterministic('atts', atts_star - tt.mean(atts_star))
        defs = pm.Deterministic('defs', defs_star - tt.mean(defs_star))

        #Make the model identifiable
        home_theta = tt.exp(intercept + atts[home_team] + defs[away_team])
        away_theta = tt.exp(intercept + atts[away_team] + defs[home_team])


        home_scored=pm.Poisson('home_scored', mu=home_theta, observed=observed_home_goals)
        away_scored=pm.Poisson('away_scored', mu=away_theta, observed=observed_away_goals)


        trace=pm.sample(3000, njobs=3)

    
    #Get the teams which are in the world cup
    inds=np.where(np.in1d(all_teams, wc_teams))
    names=all_teams[np.where(np.in1d(all_teams, wc_teams))]

    #Knockout round where England lose to Belgium
    knockout_teams_1=['Uruguay', 'Portugal', 'France', 'Argentina', 'Brazil', 'Mexico', 'Belgium', 'Japan', 'Spain', 'Russia', 'Croatia', 'Denmark', 'Sweden', 'Switzerland', 'Colombia', 'England']

    #Kncokout round where we win the group
    knockout_teams_2=['Uruguay', 'Portugal', 'France', 'Argentina', 'Brazil', 'Mexico', 'England', 'Japan', 'Spain', 'Russia', 'Croatia', 'Denmark', 'Sweden', 'Switzerland', 'Colombia', 'Belgium']


    #Do the simulations
    n_sims=10000
    f, winner=world_cup(knockout_teams_1)
    for i in tqdm(range(n_sims)):
        b, _=world_cup(knockout_teams_1)

        #gs=pd.concat((gs, a), axis=1)
        f=pd.concat((f, b), axis=1)


    #get the results as precentages
    res=f.T.apply(pd.Series.value_counts)/n_sims*100.
    f.to_csv('sims_correct_WC_bracket.csv')
    res.to_csv('sims_correct_WC_bracket_summary.csv')
    #res.T.sort_values(4.0, ascending=False)


    #Plot things
    fig, ax=plt.subplots(figsize=(17, 8))
    cax=ax.matshow(res, cmap='viridis')
    ax.set_xticks(np.arange(16))
    ax.set_xticklabels(['']+res.columns, rotation=45, fontsize=20)
    ax.set_yticklabels(['']+['Last 16', 'QF', 'SF', 'F', 'Winner'], fontsize=20)

    for (i, j), z in np.ndenumerate(res):
        ax.text(j, i, '{:0.1f}%'.format(z), ha='center', va='center', fontsize=10, color='k', bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))

    fig.tight_layout()

    
    #make violin plots

    #Sort by which team are best at attacking...
    medians=np.percentile(trace['atts'], 50, axis=0)
    top_inds=np.argsort(medians)[-10:]
    bottom_inds=np.argsort(medians)[:10]
    sorted_inds=np.concatenate((bottom_inds, top_inds))


    t=trace['atts'][:, sorted_inds]
    n=all_teams[sorted_inds]
    n=np.insert(n, 10, [''])

    fig, ax=make_violinplot(t, n, sorted_inds, cmap='plasma', context='seaborn-white', title='Attacking', xlabel='Attack')

    #...and defending
    medians=np.percentile(trace['defs'], 50, axis=0)
    top_inds=np.argsort(medians)[-10:]
    bottom_inds=np.argsort(medians)[:10]
    sorted_inds=np.concatenate((bottom_inds, top_inds))

    t=trace['defs'][:, sorted_inds]
    n=all_teams[sorted_inds]
    n=np.insert(n, 10, [''])

    fig, ax=make_violinplot(t, n, sorted_inds, cmap='plasma', context='seaborn-white', title='Defending', xlabel='Defend')

