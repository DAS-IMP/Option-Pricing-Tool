import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from streamlit.components.v1 import html
from streamlit_extras.stylable_container import stylable_container
from scipy.stats import norm
from matplotlib.colors import LinearSegmentedColormap


st.set_page_config(
    page_title="Option Pricing Tool - by Don-Angelo Sfeir",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': '''### Multi Model Option Pricing Tool. 
Please feel free to send me comments and feedback so that I can further improve the tool.

[Linkedin](https://www.linkedin.com/in/don-angelo-sfeir/)

[E-mail](mailto:sfeirdonangelo@gmail.com)'''
    }
)


def open_page(url):
    open_script= """
        <script type="text/javascript">
            window.open('%s', '_blank').focus();
        </script>
    """ % (url)
    html(open_script)



def CallBox(call_price):
  

    with stylable_container(
        key = "call_container",
        css_styles="""
            {
                border: none;
                border-radius: 20px;
                padding: 0px;
                background-color: green;
                color: black;
                text-align: center;
            }
            """,
    ):
        
        display_price = "$"+str(call_price)
        
        st.markdown(f'<h1 style="color:black;">{"Call Price"}</h1>', unsafe_allow_html=True)
        st.markdown('<h2 style="color:black;">%s</h2>' %(display_price), unsafe_allow_html=True)


def PutBox(put_price):
  

    with stylable_container(
        key = "put_container",
        css_styles="""
            {
                border: none;
                border-radius: 20px;
                padding: 0px;
                background-color: red;
                color: black;
                text-align: center;
            }
            """,
    ):
                
        display_price = "$"+str(put_price)
        
        st.markdown('<h1 style="color:black;">Put Price</h1>', unsafe_allow_html=True)
        st.markdown('<h2 style="color:black;">%s</h2>' %(display_price), unsafe_allow_html=True)
        

def CallBox_pnl(call_price):
  

    with stylable_container(
        key = "call_container",
        css_styles="""
            {
                border: none;
                border-radius: 20px;
                padding: 0px;
                background-color: green;
                color: black;
                text-align: center;
            }
            """,
    ):
        
        display_price = '{0:.5}'.format("$"+str(call_price))
        
        
        call_price = int(call_price)
        
        
        st.markdown(f'<h1 style="color:black;">{"Call PnL"}</h1>', unsafe_allow_html=True)
        if call_price > 0:
            st.markdown('<h2 style="color:black;">%s</h2>' %(display_price), unsafe_allow_html=True)
        else:
            st.markdown('<h2 style="color:black;">(%s)</h2>' %(display_price), unsafe_allow_html=True)


def PutBox_pnl(put_price):
  

    with stylable_container(
        key = "put_container",
        css_styles="""
            {
                border: none;
                border-radius: 20px;
                padding: 0px;
                background-color: red;
                color: black;
                text-align: center;
            }
            """,
    ):
                
        display_price = '{0:.5}'.format("$"+str(put_price))
        
        put_price = float(put_price)
        
        
        st.markdown(f'<h1 style="color:black;">{"Put PnL"}</h1>', unsafe_allow_html=True)
        if put_price > 0:
            st.markdown('<h2 style="color:black;">%s</h2>' %(display_price), unsafe_allow_html=True)
        else:
            st.markdown('<h2 style="color:black;">(%s)</h2>' %(display_price), unsafe_allow_html=True)



def black_scholes(S, K, T, r, sigma):
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
                
        return call_price, put_price
    
def binomial(S, K, T, r, sigma, N):
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)

    # Initialize asset prices at maturity
    ST = np.array([S * (u ** (N - i)) * (d ** i) for i in range(N + 1)])

    # Initialize option values at maturity
    C_call = np.maximum(ST - K, 0)
    C_put = np.maximum(K - ST, 0)

    # Backward induction through the tree
    for i in range(N - 1, -1, -1):
        C_call = np.exp(-r * dt) * (p * C_call[:-1] + (1 - p) * C_call[1:])
        C_put = np.exp(-r * dt) * (p * C_put[:-1] + (1 - p) * C_put[1:])

    return C_call[0], C_put[0]

def calculate_greeks(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    
    delta_call = norm.cdf(d1)
    gamma_call = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    theta_call = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)
    vega_call = S * norm.pdf(d1) * np.sqrt(T)
    rho_call = K * T * np.exp(-r * T) * norm.cdf(d2)
    
    
    delta_put = -norm.cdf(-d1)
    gamma_put = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    theta_put = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)
    vega_put = S * norm.pdf(d1) * np.sqrt(T)
    rho_put = -K * T * np.exp(-r * T) * norm.cdf(-d2)
    
    deltas = ['{0:.2f}'.format(delta_call), '{0:.2f}'.format(delta_put)]
    gammas = ['{0:.2f}'.format(gamma_call), '{0:.2f}'.format(gamma_put)]
    thetas = ['{0:.2f}'.format(theta_call), '{0:.2f}'.format(theta_put)]
    vegas = ['{0:.2f}'.format(vega_call), '{0:.2f}'.format(vega_put)]
    rhos = ['{0:.2f}'.format(rho_call), '{0:.2f}'.format(rho_put)]
    
    table_setup = {'Delta': deltas, 'Gamma': gammas, 'Theta': thetas, 'Vega': vegas, 'Rho': rhos}
    
    greek_table = pd.DataFrame(data = table_setup, index = ["Call", 'Put'])

    return greek_table, d1, d2, delta_call, delta_put, gamma_call, gamma_put, theta_call, theta_put, vega_call, rho_call, rho_put

def volga(vega, sigma, d1, d2):
    volga = vega / sigma * d1 * d2
    return volga

def vanna(vega, S, T, sigma, d2):
    vanna = -vega / (S * sigma * np.sqrt(T)) * d2
    return vanna



def sensitivity_analysis(spot, volatility, spot_sense, volatility_sense, strike_price, time_to_maturity, interest_rate_decimal):
    
    sense_call = [[0 for i in range(9)] for j in range(9)]
    sense_put = [[0 for i in range(9)] for j in range(9)]
    
    spot_sense_step = spot_sense/800
    spot_row = [ spot * ((1-(spot_sense/100))+i*2*spot_sense_step) for i in range(9)]
    
    volatility_sense_step = volatility_sense/800
    volatility_column = [ volatility * ((1-(volatility_sense/100))+i*2*volatility_sense_step) for i in range(9)]

    for i in range(len(volatility_column)):
        for j in range(len(spot_row)):
            sense_call[i][j], sense_put[i][j] = black_scholes(spot_row[j], strike_price, time_to_maturity, interest_rate_decimal, volatility_column[i]) 
    
    display_values_row = ['{0:.2f}'.format(spot_row[i]) for i in range(9)]
    display_values_column = ['{0:.4f}'.format(volatility_column[i]) for i in range(9)]    
    
    final_product_call = pd.DataFrame(sense_call, columns = display_values_column, index = display_values_row)
    final_product_put = pd.DataFrame(sense_put, columns = display_values_column, index = display_values_row)
    
    return final_product_call, final_product_put, display_values_column, display_values_row

def sensitivity_analysis_binomial(spot, volatility, spot_sense, volatility_sense, strike_price, time_to_maturity, interest_rate_decimal, steps):
    
    sense_call = [[0 for i in range(9)] for j in range(9)]
    sense_put = [[0 for i in range(9)] for j in range(9)]
    
    spot_sense_step = spot_sense/800
    spot_row = [ spot * ((1-(spot_sense/100))+i*2*spot_sense_step) for i in range(9)]
    
    volatility_sense_step = volatility_sense/800
    volatility_column = [ volatility * ((1-(volatility_sense/100))+i*2*volatility_sense_step) for i in range(9)]

    for i in range(len(volatility_column)):
        for j in range(len(spot_row)):
            sense_call[i][j], sense_put[i][j] = binomial(spot_row[j], strike_price, time_to_maturity, interest_rate_decimal, volatility_column[i], steps) 
    
    
    display_values_row = ['{0:.2f}'.format(spot_row[i]) for i in range(9)]
    display_values_column = ['{0:.4f}'.format(volatility_column[i]) for i in range(9)]
    
    final_product_call = pd.DataFrame(sense_call, columns = display_values_column, index = display_values_row)
    final_product_put = pd.DataFrame(sense_put, columns = display_values_column, index = display_values_row)
    
    return final_product_call, final_product_put, display_values_column, display_values_row

def generate_colored_dataframe(pdframe, row_display, column_display, tool_type):
    
    # Get the central value
    central_value = pdframe.iloc[4, 4]
    
    # Calculate the difference from the central value
    if tool_type == 'Option Pricing':
        diff = pdframe - central_value
    else:
        diff = pdframe
    
    # Normalize the difference to use for color intensity
    max_diff = max(abs(diff.min().min()), abs(diff.max().max()))
    norm_diff = diff / max_diff

    colours = [(0, 'red'), (0.5, 'yellow'), (1, 'green')]
    cust_cmap = LinearSegmentedColormap.from_list('custom map', colours)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(norm_diff, cmap = cust_cmap, vmin=-1, vmax=1)

    # Add text annotations
    for i in range(9):
        for j in range(9):
            if pdframe.iloc[i,j] > 0:
                text = ax.text(j, i, '{0:.2f}'.format(pdframe.iloc[i, j]),
                           ha="center", va="center", color="black")
            else:
                text = ax.text(j, i, '({0:.2f})'.format(abs(pdframe.iloc[i, j])),
                           ha="center", va="center", color="black")

    # Remove axes ticks
    ax.set_xticks(np.arange(9), labels = row_display)
    ax.set_yticks(np.arange(9), labels = column_display)
    plt.xlabel('Implied Volatility')
    plt.ylabel('Spot Price')

    return fig








st.sidebar.write('''# Options Pricing Tool\
                By:''')

st.sidebar.button('Don-Angelo Sfeir', on_click=open_page, args=('https://www.linkedin.com/in/don-angelo-sfeir/',))

"After completeing input close side-bar (top right of side-bar) for best view of the analysis."

Model_Selection = st.sidebar.radio("Choose Pricing Model", ["Black-Scholes", "Binomial"])

if Model_Selection == "Black-Scholes":
    Tool_selection = st.sidebar.radio('Choose Task', ["Option Pricing", 'PnL Analysis'])
    if Tool_selection == 'Option Pricing':
        current_asset_price = st.sidebar.number_input("Current Asset Price($)", min_value = 0.00, value = 100.00, key = 'spot input')
        strike_price = st.sidebar.number_input("Strike Price($)", min_value = 0.00, value = 100.00, key = "strike input")
        time_to_maturity = st.sidebar.number_input("Time to Maturity (years)", min_value = 0.00, value = 1.00, key = 'time to maturity input')
        volatility = st.sidebar.number_input("Implied Volatility(σ)", min_value = 0.00, value = 0.20, key = 'volatility input')
        interest_rate_percent = st.sidebar.number_input("Risk Free Interest Rate(%)", min_value = 0.00, value = 5.00, key = 'risk free interest rate input')
                        
        st.sidebar.write("---")
        st.sidebar.write("## Heat Map Parameters")
        
        spot_heatmap_spread = st.sidebar.number_input("Spot Sensitivity(%)", value = 10.00, min_value = 0.00, max_value = 100.00, key = "spot price sens %")
        spot_heatmap_step = spot_heatmap_spread/900
        
        vol_heatmap_spread = st.sidebar.number_input("Implied Volatility Sensitivity(%)", value = 10.00, min_value = 0.00, max_value = 100.00, key = "vol sens %")
        vol_heatmap_step = vol_heatmap_spread/900
    
        interest_rate_decimal = interest_rate_percent/100
        
        call_price, put_price = black_scholes(current_asset_price, strike_price, time_to_maturity, interest_rate_decimal, volatility)
    
        call_price = '{0:.2f}'.format(call_price)
        put_price = '{0:.2f}'.format(put_price)
        
        st.title("Black-Scholes Model")
        
        input_display = pd.DataFrame([['{0:.2f}'.format(current_asset_price), '{0:.2f}'.format(strike_price), '{0:.2f}'.format(time_to_maturity), '{0:.2f}'.format(volatility), '{0:.2f}'.format(interest_rate_percent)]],columns = ["Current Asset Price($)", "Strike Price($)", "Time to Maturity (years)", "Implied Volatility(σ)", "Risk Free Interest Rate(%)"])
        st.table(input_display)
        
        call_col, put_col = st.columns(2)
        
        with call_col:
            CallBox(call_price)
        with put_col:
            PutBox(put_price)
        
        greek_table, d1, d2, delta_call, delta_put, gamma_call, gamma_put, theta_call, theta_put, vega_call, rho_call, rho_put = calculate_greeks(current_asset_price, strike_price, time_to_maturity, interest_rate_decimal, volatility)
        
        st.table(greek_table)
        
        st.write('---')
        
        st.title("Value Sensitivity Analysis")
        st.write(' ')
        
        call_sense_data, put_sense_data, column_display, row_display =sensitivity_analysis(current_asset_price, volatility, spot_heatmap_spread, vol_heatmap_spread, strike_price, time_to_maturity, interest_rate_decimal)    
        
        call_sense_col, put_sense_col = st.columns(2)
        
        with call_sense_col:
            st.write('### Call Price Sensitivity')
            st.write(' ')
            fig = generate_colored_dataframe(call_sense_data, column_display, row_display, Tool_selection)
            st.pyplot(fig)        
        with put_sense_col:
            st.write('### Put Price Sensitivity')
            st.write(' ')
            fig = generate_colored_dataframe(put_sense_data, column_display, row_display, Tool_selection)
            st.pyplot(fig)    
        
    else:
        purchase_price = st.sidebar.number_input("Price Paid for Option($)", min_value = 0.00, value = 5.00, key = 'purchase input')
        current_asset_price = st.sidebar.number_input("Current Asset Price($)", min_value = 0.00, value = 100.00, key = 'spot input')
        strike_price = st.sidebar.number_input("Strike Price($)", min_value = 0.00, value = 100.00, key = "strike input")
        time_to_maturity = st.sidebar.number_input("Time to Maturity (years)", min_value = 0.00, value = 1.00, key = 'time to maturity input')
        volatility = st.sidebar.number_input("Implied Volatility(σ)", min_value = 0.00, value = 0.20, key = 'volatility input')
        interest_rate_percent = st.sidebar.number_input("Risk Free Interest Rate(%)", min_value = 0.00, value = 5.00, key = 'risk free interest rate input') 
                
        st.sidebar.write("---")
        
        st.sidebar.write("## Heat Map Parameters")
        
        spot_heatmap_spread = st.sidebar.number_input("Spot Sensitivity(%)", value = 10.00, min_value = 0.00, max_value = 100.00, key = "spot price sens %")
        spot_heatmap_step = spot_heatmap_spread/900
        
        vol_heatmap_spread = st.sidebar.number_input("Implied Volatility Sensitivity(%)", value = 10.00, min_value = 0.00, max_value = 100.00, key = "vol sens %")
        vol_heatmap_step = vol_heatmap_spread/900
        
        interest_rate_decimal = interest_rate_percent/100
        
        st.sidebar.write("---")
        st.sidebar.write("## PnL Decomposition Parameters")
        
        delta_volatility = st.sidebar.number_input("Change in Implied Volatility(σ)", value = 0.10, key = 'change in volatility input')
        realised_volatility = st.sidebar.number_input("Realised Volatility(σ)", min_value = 0.00, value = 0.20, key = 'realised volatility input')
        dt = st.sidebar.number_input("Number of Trading Days in the Future", min_value = 1, value = 1, key = 'dt input')/252
        
        call_price, put_price = black_scholes(current_asset_price, strike_price, time_to_maturity, interest_rate_decimal, volatility)
        
        call_pnl = call_price - purchase_price
        put_pnl = put_price - purchase_price
        
        call_price = '{0:.2f}'.format(call_price)
        put_price = '{0:.2f}'.format(put_price)
        
        st.title("Black-Scholes Model")
        
        input_display = pd.DataFrame([['{0:.2f}'.format(current_asset_price), '{0:.2f}'.format(strike_price), '{0:.2f}'.format(time_to_maturity), '{0:.2f}'.format(volatility), '{0:.2f}'.format(interest_rate_percent)]],columns = ["Current Asset Price($)", "Strike Price($)", "Time to Maturity (years)", "Implied Volatility(σ)", "Risk Free Interest Rate(%)"])
        st.table(input_display)
        
        call_col, put_col = st.columns(2)
        
        with call_col:
            CallBox_pnl(call_pnl)
        with put_col:
            PutBox_pnl(put_pnl)
        
        greek_table, d1, d2, delta_call, delta_put, gamma_call, gamma_put, theta_call, theta_put, vega, rho_call, rho_put = calculate_greeks(current_asset_price, strike_price, time_to_maturity, interest_rate_decimal, volatility)
        
        st.table(greek_table)        
        
        st.write('---')
        
        st.title("PnL Sensitivity Analysis (Price of Option)")
        st.write(' ')
        
        call_sense_data, put_sense_data, column_display, row_display =sensitivity_analysis(current_asset_price, volatility, spot_heatmap_spread, vol_heatmap_spread, strike_price, time_to_maturity, interest_rate_decimal)
        
        call_sense_data = round(call_sense_data - purchase_price, 2)
        put_sense_data = round(put_sense_data - purchase_price, 2)
        
        call_sense_col, put_sense_col = st.columns(2)
        
        with call_sense_col:
            st.write('### Call Price Sensitivity')
            st.write(' ')
            fig = generate_colored_dataframe(call_sense_data, column_display, row_display, Tool_selection)
            st.pyplot(fig)        
        with put_sense_col:
            st.write('### Put Price Sensitivity')
            st.write(' ')
            fig = generate_colored_dataframe(put_sense_data, column_display, row_display, Tool_selection)
            st.pyplot(fig)        
        
        
        dS = current_asset_price * realised_volatility * np.sqrt(dt)
        T1 = time_to_maturity - dt
        S1_call = current_asset_price + dS
        S1_put = current_asset_price - dS
        sigma1 = volatility + delta_volatility
        
        
        
        call_price_future, fill = black_scholes(S1_call, strike_price, T1, interest_rate_decimal, sigma1)
        fill2, put_price_future = black_scholes(S1_put, strike_price, T1, interest_rate_decimal, sigma1)
        
        call_dPandL = round(call_price_future - float(call_price) - delta_call * dS, 2)
        put_dPandL = round(put_price_future - float(put_price) + delta_put * dS, 2)
        
        st.title('Intra-Period Decomposed PnL (Delta-Hedged)')
        st.write(' ')
        
        decomposed_call_col, decomposed_put_col = st.columns(2)
        
        with decomposed_call_col:
            CallBox_pnl(call_dPandL)
            
            delta_PandL = 0
            theta_PandL = theta_call * dt
            vega_PandL = vega * delta_volatility
            gamma_PandL = 1 / 2 * gamma_call * dS**2
            volga_PandL = 1 / 2 * volga(vega, volatility, d1, d2) * delta_volatility**2
            vanna_PandL = vanna(vega, current_asset_price, time_to_maturity, volatility, d2) * dS * delta_volatility
            unexplained = call_dPandL - sum([delta_PandL, theta_PandL, vega_PandL, gamma_PandL, volga_PandL, vanna_PandL])
            
            y = [delta_PandL, theta_PandL, vega_PandL, gamma_PandL, volga_PandL, vanna_PandL, unexplained]
            x = ["Delta", "Theta", "Vega", "Gamma", "Volga", "Vanna","Higher-Order Terms"]
            
            fig = plt.figure(figsize=(10, 3))
            plt.grid(zorder = 0)
            plt.bar(x, y, zorder = 5)
            
            plt.title("P&L Decomposition")
            plt.show();
            st.pyplot(fig)
            
            data = [[delta_PandL, theta_PandL, vega_PandL, gamma_PandL, volga_PandL, vanna_PandL, unexplained]]
            index = ["Delta", "Theta", "Vega", "Gamma", "Volga", "Vanna","HOTs"]
            
            st.table(pd.DataFrame(data, columns = index, index = [""]))

            
            
        with decomposed_put_col:
            PutBox_pnl(put_dPandL)

            delta_PandL = 0
            theta_PandL = theta_put * dt
            vega_PandL = vega * delta_volatility
            gamma_PandL = 1 / 2 * gamma_put * dS**2
            volga_PandL = 1 / 2 * volga(vega, volatility, d1, d2) * delta_volatility**2
            vanna_PandL = vanna(vega, current_asset_price, time_to_maturity, volatility, d2) * dS * delta_volatility
            unexplained = put_dPandL - sum([delta_PandL, theta_PandL, vega_PandL, gamma_PandL, volga_PandL, vanna_PandL])
            
            y = [delta_PandL, theta_PandL, vega_PandL, gamma_PandL, volga_PandL, vanna_PandL, unexplained]
            x = ["Delta", "Theta", "Vega", "Gamma", "Volga", "Vanna","Higher-Order Terms"]
            
            data = [[delta_PandL, theta_PandL, vega_PandL, gamma_PandL, volga_PandL, vanna_PandL, unexplained]]
            index = ["Delta", "Theta", "Vega", "Gamma", "Volga", "Vanna","HOTs"]
            
            fig = plt.figure(figsize=(10, 3))
            plt.grid(zorder = 0)
            plt.bar(x, y, zorder = 5)
            
            plt.title("P&L Decomposition")
            plt.show();
            st.pyplot(fig)                                    
            
            st.table(pd.DataFrame(data, columns = index, index = [""]))



if Model_Selection == "Binomial":
    Tool_selection = st.sidebar.radio('Choose Task', ["Option Pricing", 'PnL Analysis'])
    if Tool_selection == 'Option Pricing':
        current_asset_price = st.sidebar.number_input("Current Asset Price($)", min_value = 0.00, value = 100.00, key = 'spot input')
        strike_price = st.sidebar.number_input("Strike Price($)", min_value = 0.00, value = 100.00, key = "strike input")
        time_to_maturity = st.sidebar.number_input("Time to Maturity (years)", min_value = 0.00, value = 1.00, key = 'time to maturity input')
        volatility = st.sidebar.number_input("Implied Volatility(σ)", min_value = 0.00, value = 0.20, key = 'volatility input')
        interest_rate_percent = st.sidebar.number_input("Risk Free Interest Rate(%)", min_value = 0.00, value = 5.00, key = 'risk free interest rate input')
        steps = st.sidebar.number_input("Number of Steps", min_value = 0, value = 100, key = 'number of steps input')
                
        st.sidebar.write("---")
        
        st.sidebar.write("## Heat Map Parameters")
        
        spot_heatmap_spread = st.sidebar.number_input("Spot Sensitivity(%)", value = 10.00, min_value = 0.00, max_value = 100.00, key = "spot price sens %")
        spot_heatmap_step = spot_heatmap_spread/900
        
        vol_heatmap_spread = st.sidebar.number_input("Implied Volatility Sensitivity(%)", value = 10.00, min_value = 0.00, max_value = 100.00, key = "vol sens %")
        vol_heatmap_step = vol_heatmap_spread/900
    
        
        interest_rate_decimal = interest_rate_percent/100
        
        call_price, put_price = binomial(current_asset_price, strike_price, time_to_maturity, interest_rate_decimal, volatility, steps)
    
        call_price = '{0:.2f}'.format(call_price)
        put_price = '{0:.2f}'.format(put_price)
        
        st.title("Binomial Model")
        
        input_display = pd.DataFrame([['{0:.2f}'.format(current_asset_price), '{0:.2f}'.format(strike_price), '{0:.2f}'.format(time_to_maturity), '{0:.2f}'.format(volatility), '{0:.2f}'.format(interest_rate_percent), steps]],columns = ["Current Asset Price($)", "Strike Price($)", "Time to Maturity (years)", "Implied Volatility(σ)", "Risk Free Interest Rate(%)", "Number of Steps"])
        st.table(input_display)
        
        call_col, put_col = st.columns(2)
        
        with call_col:
            CallBox(call_price)
        with put_col:
            PutBox(put_price)
        
        greek_table, d1, d2, delta_call, delta_put, gamma_call, gamma_put, theta_call, theta_put, vega_call, rho_call, rho_put = calculate_greeks(current_asset_price, strike_price, time_to_maturity, interest_rate_decimal, volatility)
        
        st.table(greek_table)
        
        st.write('---')
        
        st.title("Value Sensitivity Analysis")
        st.write(' ')
        
        call_sense_data, put_sense_data, column_display, row_display =sensitivity_analysis_binomial(current_asset_price, volatility, spot_heatmap_spread, vol_heatmap_spread, strike_price, time_to_maturity, interest_rate_decimal, steps)
        
        
        call_sense_col, put_sense_col = st.columns(2)
        
        with call_sense_col:
            st.write('### Call Price Sensitivity')
            st.write(' ')
            fig = generate_colored_dataframe(call_sense_data, column_display, row_display, Tool_selection)
            st.pyplot(fig)        
        with put_sense_col:
            st.write('### Put Price Sensitivity')
            st.write(' ')
            fig = generate_colored_dataframe(put_sense_data, column_display, row_display, Tool_selection)
            st.pyplot(fig)    
        
    else:
        purchase_price = st.sidebar.number_input("Price Paid for Option($)", min_value = 0.00, value = 5.00, key = 'purchase input')
        current_asset_price = st.sidebar.number_input("Current Asset Price($)", min_value = 0.00, value = 100.00, key = 'spot input')
        strike_price = st.sidebar.number_input("Strike Price($)", min_value = 0.00, value = 100.00, key = "strike input")
        time_to_maturity = st.sidebar.number_input("Time to Maturity (years)", min_value = 0.00, value = 1.00, key = 'time to maturity input')
        volatility = st.sidebar.number_input("Implied Volatility(σ)", min_value = 0.00, value = 0.20, key = 'volatility input')
        interest_rate_percent = st.sidebar.number_input("Risk Free Interest Rate(%)", min_value = 0.00, value = 5.00, key = 'risk free interest rate input')
        steps = st.sidebar.number_input("Number of Steps", min_value = 0, value = 100, key = 'number of steps input')        
                
        st.sidebar.write("---")
        
        st.sidebar.write("## Heat Map Parameters")
        
        spot_heatmap_spread = st.sidebar.number_input("Spot Sensitivity(%)", value = 10.00, min_value = 0.00, max_value = 100.00, key = "spot price sens %")
        spot_heatmap_step = spot_heatmap_spread/900
        
        vol_heatmap_spread = st.sidebar.number_input("Implied Volatility Sensitivity(%)", value = 10.00, min_value = 0.00, max_value = 100.00, key = "vol sens %")
        vol_heatmap_step = vol_heatmap_spread/900
    
        
        interest_rate_decimal = interest_rate_percent/100
        
        st.sidebar.write("---")
        st.sidebar.write("## PnL Decomposition Parameters")
        
        delta_volatility = st.sidebar.number_input("Change in Implied Volatility(σ)", value = 0.10, key = 'change in volatility input')
        realised_volatility = st.sidebar.number_input("Realised Volatility(σ)", min_value = 0.00, value = 0.20, key = 'realised volatility input')
        dt = st.sidebar.number_input("Number of Trading Days in the Future", min_value = 1, value = 1, key = 'dt input')/252
        
        
        call_price, put_price = binomial(current_asset_price, strike_price, time_to_maturity, interest_rate_decimal, volatility, steps)
        
        call_pnl = call_price - purchase_price
        put_pnl = put_price - purchase_price
        
        call_price = '{0:.2f}'.format(call_price)
        put_price = '{0:.2f}'.format(put_price)
        
        
        st.title("Binomial Model")
        
        input_display = pd.DataFrame([['{0:.2f}'.format(current_asset_price), '{0:.2f}'.format(strike_price), '{0:.2f}'.format(time_to_maturity), '{0:.2f}'.format(volatility), '{0:.2f}'.format(interest_rate_percent), steps]],columns = ["Current Asset Price($)", "Strike Price($)", "Time to Maturity (years)", "Implied Volatility(σ)", "Risk Free Interest Rate(%)", "Number of Steps"])
        
        st.table(input_display)
        
        call_col, put_col = st.columns(2)
        
        with call_col:
            CallBox_pnl(call_pnl)
        with put_col:
            PutBox_pnl(put_pnl)
        
        greek_table, d1, d2, delta_call, delta_put, gamma_call, gamma_put, theta_call, theta_put, vega, rho_call, rho_put = calculate_greeks(current_asset_price, strike_price, time_to_maturity, interest_rate_decimal, volatility)
        
        st.table(greek_table)
        
        st.write('---')
        
        st.title("PnL Sensitivity Analysis (Price of Option)")
        st.write(' ')
        
        call_sense_data, put_sense_data, column_display, row_display =sensitivity_analysis(current_asset_price, volatility, spot_heatmap_spread, vol_heatmap_spread, strike_price, time_to_maturity, interest_rate_decimal)
        
        call_sense_data = round(call_sense_data - purchase_price, 2)
        put_sense_data = round(put_sense_data - purchase_price, 2)
        
        call_sense_col, put_sense_col = st.columns(2)
        
        with call_sense_col:
            st.write('### Call Price Sensitivity')
            st.write(' ')
            fig = generate_colored_dataframe(call_sense_data, column_display, row_display, Tool_selection)
            st.pyplot(fig)        
        with put_sense_col:
            st.write('### Put Price Sensitivity')
            st.write(' ')
            fig = generate_colored_dataframe(put_sense_data, column_display, row_display, Tool_selection)
            st.pyplot(fig)
            
        dS = current_asset_price * realised_volatility * np.sqrt(dt)
        T1 = time_to_maturity - dt
        S1_call = current_asset_price + dS
        S1_put = current_asset_price - dS
        sigma1 = volatility + delta_volatility
        
        
        
        call_price_future, fill = binomial(S1_call, strike_price, T1, interest_rate_decimal, sigma1, steps)
        fill2, put_price_future = binomial(S1_put, strike_price, T1, interest_rate_decimal, sigma1, steps)
        
        call_dPandL = round(call_price_future - float(call_price) - delta_call * dS, 2)
        put_dPandL = round(put_price_future - float(put_price) + delta_put * dS, 2)
        
        st.title('Intra-Period Decomposed PnL (Delta-Hedged)')
        st.write(' ')
        
        decomposed_call_col, decomposed_put_col = st.columns(2)
        
        with decomposed_call_col:
            CallBox_pnl(call_dPandL)
            
            delta_PandL = 0
            theta_PandL = theta_call * dt
            vega_PandL = vega * delta_volatility
            gamma_PandL = 1 / 2 * gamma_call * dS**2
            volga_PandL = 1 / 2 * volga(vega, volatility, d1, d2) * delta_volatility**2
            vanna_PandL = vanna(vega, current_asset_price, time_to_maturity, volatility, d2) * dS * delta_volatility
            unexplained = call_dPandL - sum([delta_PandL, theta_PandL, vega_PandL, gamma_PandL, volga_PandL, vanna_PandL])
            
            y = [delta_PandL, theta_PandL, vega_PandL, gamma_PandL, volga_PandL, vanna_PandL, unexplained]
            x = ["Delta", "Theta", "Vega", "Gamma", "Volga", "Vanna","Higher-Order Terms"]
            
            fig = plt.figure(figsize=(10, 3))
            plt.grid(zorder = 0)
            plt.bar(x, y, zorder = 5)
            
            plt.title("P&L Decomposition")
            plt.show();
            st.pyplot(fig)
            
            data = [[delta_PandL, theta_PandL, vega_PandL, gamma_PandL, volga_PandL, vanna_PandL, unexplained]]
            index = ["Delta", "Theta", "Vega", "Gamma", "Volga", "Vanna","HOTs"]
            
            st.table(pd.DataFrame(data, columns = index, index = [""]))

            
        with decomposed_put_col:
            PutBox_pnl(put_dPandL)

            delta_PandL = 0
            theta_PandL = theta_put * dt
            vega_PandL = vega * delta_volatility
            gamma_PandL = 1 / 2 * gamma_put * dS**2
            volga_PandL = 1 / 2 * volga(vega, volatility, d1, d2) * delta_volatility**2
            vanna_PandL = vanna(vega, current_asset_price, time_to_maturity, volatility, d2) * dS * delta_volatility
            unexplained = put_dPandL - sum([delta_PandL, theta_PandL, vega_PandL, gamma_PandL, volga_PandL, vanna_PandL])
            
            y = [delta_PandL, theta_PandL, vega_PandL, gamma_PandL, volga_PandL, vanna_PandL, unexplained]
            x = ["Delta", "Theta", "Vega", "Gamma", "Volga", "Vanna","Higher-Order Terms"]
            
            fig = plt.figure(figsize=(10, 3))
            plt.grid(zorder = 0)
            plt.bar(x, y, zorder = 5)
            
            plt.title("P&L Decomposition")
            plt.show();
            st.pyplot(fig)                                    
            
            
            data = [[delta_PandL, theta_PandL, vega_PandL, gamma_PandL, volga_PandL, vanna_PandL, unexplained]]
            index = ["Delta", "Theta", "Vega", "Gamma", "Volga", "Vanna","HOTs"]
            
            st.table(pd.DataFrame(data, columns = index, index = [""]))
