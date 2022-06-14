import streamlit as st
import joblib
import pandas as pd
st.write("# Internet Firewall Prediction")



col1, col2, col3 = st.columns(3)

Source_Port= col1.number_input("Enter Source Port")

Destination_Port = col2.number_input("Enter Destination Port")
NAT_Source_Port= col3.number_input("Enter NAT Source Port")

NAT_Destination_Port = col1.number_input("Enter  NAT Destination Port")

Bytes = col2.number_input("Enter No. of Bytes")

Bytes_Sent = col3.number_input("Bytes Sent ")

Bytes_Recieved = col1.number_input("Bytes Recieved")

Packets = col2.number_input("Packets")

Elapsed_Time = col3.number_input("Elapsed Time in sec")

pkts_sent = col1.number_input("Packets Sent")

pkts_recieved = col2.number_input("Packets Recieved")


x = st.button('Predict')



df_pred = pd.DataFrame([[Source_Port , Destination_Port , NAT_Source_Port , NAT_Destination_Port , Bytes, Bytes_Sent , Bytes_Recieved ,Packets, Elapsed_Time,pkts_sent,pkts_recieved]] ,

columns = ['Source Port' , 'Destination Port' , 'NAT Source Port' , ' NAT Destination Port' , 'Bytes' , 'Bytes Sent' , 'Bytes Recieved' , 'Packets' , 'Elapsed Time (sec)' , 'pkts_sent' , 'pkts_recieved'])



model = joblib.load('savedmodel.pkl')


prediction = model.predict(df_pred)

print(prediction)
if(prediction[0]=='allow'):
        st.write('<p class="big-font">Allow.</p>',unsafe_allow_html=True)
elif(prediction[0]=='deny'):
        st.write('<p class="big-font">Deny.</p>',unsafe_allow_html=True)
elif(prediction[0]=='drop'):
        st.write('<p class="big-font">Drop.</p>',unsafe_allow_html=True)
elif(prediction[0]=='reset-both'):
        st.write('<p class="big-font">Reset Both .</p>',unsafe_allow_html=True)