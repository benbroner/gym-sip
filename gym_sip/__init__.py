from gym.envs.registration import register

register(
    id='Sip-v0',
    entry_point='gym_sip.envs.sip_env:SipEnv',
    kwargs={'file_name': 'nba2'
            #
            # , dtypes: {
            #         'a_team': category,
            #         'h_team': category,
            #         'league': category,
            #         'game_id': 'Int64',
            #         'a_pts': 'Int16',
            #         'h_pts': 'Int16',
            #         'secs': 'Int16',
            #         'status': 'Int16',
            #         'a_win': 'Int16',
            #         'h_win': 'Int16',
            #         'last_mod_to_start': 'Float64',
            #         'num_markets': 'Int16',
            #         'a_odds_ml': 'Int32',
            #         'h_odds_ml': 'Int32',
            #         'a_hcap_tot': 'Int32',
            #         'h_hcap_tot': 'Int32'
            #             }
            }
)
