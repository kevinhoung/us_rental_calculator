import streamlit as st

def show_sidebar():
    with st.sidebar:
        try:
            st.image("images/logo.png", use_container_width=True)
        except:
            st.header("Pricision AI")
        
        st.markdown("---")
        
        # --- ABOUT THE DEVELOPER (moved to top) ---
        st.markdown("### 👨‍💻 About the Developer")
        
        # --- PROFILE IMAGE SECTION ---
        # Using columns to center the image effectively
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            try:
                # Make sure your file is named 'profile.jpeg' and in the images folder
                st.image("images/profile.jpeg", use_container_width=True) 
            except:
                st.info("Add `profile.jpeg` to images folder")

        st.write(
            """
            **Kevin Houng**
            *Avid Buidler. Recovering Attorney. HR Technologist.*
            """
        )
        
        st.caption("BUIDL through Fear. 🚀")
        st.markdown("[LinkedIn](https://www.linkedin.com/in/kevinhoung/) | [GitHub](https://github.com/kevinhoung)")
            
        st.markdown("---")

        # --- THE FOUNDER STORY ---
        st.markdown("### 👨‍💻 Jack of All Trades, Master of None")
        
        with st.expander("BUIDL through Fear", expanded=True):
            st.write(
                """
                **Hi, I'm Kevin.** **I've never chosen a linear path because the only problems worth solving are the ones that scare you the most.**
                
                **The recovering litigator.**
                I started at **Columbia Law**. I wore the suit. I billed the hours. I was on the "perfect" path to partner at a top firm. 
                But I had a secret: *I dreaded going to work.* When I finally quit, a senior partner looked me dead in the eye and said, *"I've been trying to leave for years... and now I'm a partner."* That was my wake-up call. I realized that **if your decisions don't scare you a little, you aren't being ambitious enough.**
                
                **The Crypto Cowboy & People Architect.**
                I pivoted hard. I dove into the chaos of Web3 and startups, leading HR for **Coin Cloud** and **Protocol Labs**. 
                
                I didn't just push paper; I built culture in the Wild West. I implemented "Paid in Bitcoin" policies when others were scared of the internet. I learned that **talent always finds a place**, and that the only thing more infectious than a bad attitude is a good one.
                
                **The DNA of a Founder.**
                Three decades ago, my mother flew 6,000 miles to a foreign land with nothing, worked until 2 AM, and built a life without regret. 
                
                **Pricision AI** is my version of that flight.
                """
            )

        st.markdown("### 🚀 The Unfair Advantage")
        st.info(
            """
            Most founders use a hammer, so everything looks like a nail.
            
            * **I know the Law:** I see the regulatory risks and structural pitfalls before they happen.
            * **I undestand real-world HR Challenges:** I understand human behavior, incentives, and what makes people *click* (and book).
            * **I have the Tech:** I built this engine to automate the intuition I spent a decade refining.
            
            Real estate isn't just data; it's **people + regulation + market forces.** I speak all three languages.
            """
        )
