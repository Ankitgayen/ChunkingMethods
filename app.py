from rich import print
from langchain_core.documents import Document
from langchain_ollama import OllamaLLM
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.runnables import RunnablePassthrough , RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker

llm = OllamaLLM(model="gemma3:latest")


## RAG

def rag(chunks,collection_name):
    vectorstore = Chroma.from_documents(
        documents=chunks,
        collection_name=collection_name,
        embedding=OllamaEmbeddings(model="nomic-embed-text")
    )
    prompt_template = """

            Answer the question based on the only on the following context:
            {context}
            Question: {question}
          
        """
    prompt = ChatPromptTemplate.from_template(prompt_template)

    chain = (
        {"context": vectorstore.as_retriever(), "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()

    )
    result = chain.invoke("Create a proper introduction")
    print(result)

## 1. Character text Splitting
print("### Character text Splitting ###")

text = """
A speaker at a battery conference once said, “The battery is a wild animal and artificial intelligence domesticates it.” A battery is illusive and does not exhibit visible changes as part of usage; it looks the same when fully charged or empty, new or old and in need of replacement. A car tire, in comparison, distorts when low on air and indicates end-of-life when the treads are worn.

The shortcomings of a battery can be summarized by these three concerns: [1] The user does not know how much runtime the pack has left; [2] the host is uncertain if the battery can satisfy the power demand; and [3] the charger must be tailored to each battery size and chemistry. The solutions are complex and the “smart” battery promises to lessen some of these deficiencies.

Battery users imagine a battery pack as being an energy storage device that resembles a fuel tank dispensing liquid fuel. For simplicity reasons, a battery can be seen as such; however, measuring stored energy from an electrochemical device is far more complex.

While an ordinary fuel gauge measures in-and-out-flowing liquid from a tank of a known size with minimal losses, a battery fuel gauge has unconfirmed definitions and only reveals the open circuit voltage (OCV), which is a fickle reflection of state-of-charge (SoC). To compound the problem, a battery is a leaky and shrinking vessel that loses energy and takes less content with each charge. As the capacity fades, the specified Ah (ampere-hours) rating no longer holds true. Nor can the fuel gauge assess the capacity by itself; the reading always shows full after recharge even if the capacity has dropped to half the specified Ah.

The simplest method to measure state-of-charge is reading voltage, but this can be inaccurate as load currents pull the voltage down during discharge. The largest challenge is the flat discharge voltage curve on most lithium and nickel-based batteries. Temperature also plays a role; heat raises the voltage and a cold ambient lowers it. Agitation by a previous charge or discharge causes further errors and the battery needs a few hours rest to neutralize. (See BU-903: How to Measure State-charge)

Most batteries for medical, military and computing devices are “smart.” This means that some level of communication occurs between the battery, the equipment and the user. The definitions of “smart” vary among manufacturers and regulatory authorities, and the most basic smart battery may contain nothing more than a chip that sets the charger to the correct charge algorithm. In the eyes of the Smart Battery System (SBS) forum, these batteries cannot be called smart. The SBS forum states that a smart battery must provide state-of-charge indications.

Safety is a key design objective and the concept behind SBS is to place system intelligence inside the battery pack. The SBS battery thus communicates with the charge management chip in a closed loop. In spite of this digital supervision, most SBS chargers also rely on analog signals from the chemical battery to terminate the charge when the battery is full. Furthermore, redundant temperature sensing is added for safety reasons.

Benchmarq was the first company to offer fuel-gauge technology in 1990. Today, many manufacturers offer integrated circuit (IC) chips in single-wire and two-wire systems, also known as System Management Bus (SMBus).

State-of-charge estimations in a smart battery commonly include coulomb counting, a theory that goes back 250 years when Charles-Augustin de Coulomb first established the “Coulomb Rule.” Figure 1 illustrates the principle of coulomb counting, measuring in-and-out flowing energy. One coulomb (1C) per second is one ampere (1A). Discharging a battery at 1A for one hour equates to 3,600C. (Not to be confused with C-rate.)

Principle of a fuel gauge based on coulomb counting
Figure 1: Principle of a fuel gauge based on coulomb counting [1]
A circuit measures the in-and-out flowing energy; the stored energy represents state-of-charge. One coulomb per second is one ampere (1A).
Coulomb counting should be flawless but errors occur. If, for example, a battery was charged for 1 hour at 1 ampere, the same amount of energy should be available on discharge, and no battery can deliver this. Inefficiencies in charge acceptance, especially towards the end of charge and particularly if fast-charged, reduces the energy efficiency. Losses also occur in storage and during discharge. The available energy is always less than what has been fed into the battery.

Single-wire Bus
The single-wire system, also known as 1-Wire, communicates through one wire at low speed. Designed by Dallas Semiconductor Corp., the 1-Wire combines data and clock into one line for transmission; the Manchester code, also known as phase coding, separates the data at the receiving end. For safety reasons, most batteries also run a separate wire for temperature sensing. Figure 2 shows the layout of a single-wire system.

Single-wire system of a “smart” battery
Figure 2: Single-wire system of a “smart” battery [1]
A single wire provides data communication. For safety reasons, most batteries also feature a separate wire for temperature sensing.
The single-wire system stores the battery code and tracks battery data that typically includes voltage, current, temperature and state-of-charge information. Because of the relatively low hardware cost, the single-wire system is attractive for price-sensitive devices such as measuring instruments, mobile phones, two-way radios, cameras and scanners.

Most single-wire systems have their own protocol and use a customized charger. The Benchmarq single-wire solution, for example, cannot measure the current directly; state-of-health (SoH) measurement is only possible when “marrying” the host to a designated battery.

System Management Bus
The System Management Bus (SMBus) represents a concerted effort to agree on one communications protocol and one set of data. Derived from I2C, the Duracell/Intel smart battery system was standardized in 1995 and consists of two separate lines for data and clock. I2C (Inter-Integrated Circuit) is a multi-master, multi-slave, single-ended, serial computer bus invented by Philips Semiconductor. Figure 3 shows the layout of the two-wire SMBus system.

Two-wire SMBus system
Figure 3: Two-wire SMBus system [1]
The SMBus works on a two-wire system using a standardized communications protocol.
This system lends itself to standardized state-of-charge and state-of-health measurements.
The philosophy behind the SMBus battery was to remove charge control from the charger and assign it to the battery. With a true SMBus system, the battery becomes the master and the charger the slave that obeys the command of the battery. This enables a universal charger to service present and future battery chemistries by applying correct charge algorithms.

During the 1990s, several standardized SMBus battery packs emerged, including the 35 and 202 (Figure 4). Manufactured by Sony, Hitachi, GP Batteries and others, these interchangeable batteries were designed to power a broad range of portable devices, such as laptops and medical instruments. The idea was solid but standardization diverged as most manufacturers began building their own packs.

To prevent unauthorized batteries from infiltrating the market, some manufacturers add a code to exclude other pack vendors. A few manufacturers go as far as to invalidate the battery when a given cycle count is reached. To avoid surprises, most of these systems inform the user of the pending end-of-life.

35 and 202 series batteries featuring SMBus
Figure 4: 35 and 202 series batteries featuring SMBus [1]Available in nickel- and lithium-based chemistries, these batteries power laptops, biomedical instruments and survey equipment.
Non-SMBus (dumb) versions with the same footprint are also available.
An SMBus battery contains permanent and temporary data. The battery manufacturer program the permanent data into the battery, which includes battery ID, battery type, manufacturer’s name, serial number and date of manufacture. The temporary data is added during use and contains cycle count, usage pattern and maintenance requirements. Some of the information is kept, while other data is renewed throughout the life of the battery. The voltage is typically measured in 1mV increments; the current resolution is 0.5mA; temperature accuracy is about ±3ºC.

SMBus Level 2 and 3 Charging
Smart battery chargers are divided into Level 1, 2 and 3. Level 1 has been discontinued because it does not provide chemistry-independent charging and it supported a single chemistry only. A Level 2 charger is fully controlled by the Smart Battery and acts as an SMBus slave, responding to voltage and current commands from the Smart Battery. Level 2 also serves as in-circuit charging, a practice that is common in laptops. Another use is a battery with a built-in charging circuit. In Level 2, battery and circuit are married to each other.

A level 3 charger can interpret commands from a Smart Battery, as is done with Level 2, and also act as master. In other words, the Level 3 charger can request charging information from the Smart Battery but it can also impose its own charging algorithm by responding to the “chemical” battery. Most industrial smart chargers are based on the hybrid type Level 3.

Some lower-cost chargers have emerged that accommodate SMBus batteries, but these may not be fully SBS compliant. Manufacturers of SMBus batteries do not endorse this shortcut because of safety concerns. Applications such as biomedical instruments, data collection devices and survey equipment lean towards Level 3 chargers with full-fledged charge protocols. Table 5 lists the advantages and limitations of the smart battery.

Advantages	
Provides state-of-charge and full charge capacity, reflecting capacity estimations.
Configures charger to the correct algorithm.
Reminds user of periodic service.
Protects battery from unauthorized use.
Limitations	
Adds 25% to the cost of a battery. (Fuel gauge ICs are in the $2-range)
Complicates the charger; most chargers for intelligent batteries are hybrid and also service non-intelligent batteries.
Requires periodic calibration.
Readout only shows state-of-charge and not actual runtime.
Table 5: Advantages and limitations of the smart battery
Simple Guidelines for Using Smart Batteries
Calibrate a smart battery by applying a full discharge and charge every 3 months or after every 40 partial cycles. Batteries with impedance tracking provide a certain amount of self-calibration.
A fuel gauge showing 100 percent SoC does not automatically assure a good battery. The capacity may have faded to 50 percent, cutting the runtime in half. A fuel gauge can give a false sense of security.
If possible, replace the battery with the same brand to avoid incompatibility issues with the device and/or charger. Always test the battery and the charger before use.
Exercise caution when using a smart battery that does not indicate state-of-charge correctly. This battery may be faulty or is not fully compatible with the equipment.

"""

## Manual Splitting
chunks =[]
chunk_size = 35
for i in range(0,len(text),chunk_size):
    chunk = text[i:i + chunk_size]
    chunks.append(chunk)

documents = [Document(page_content=chunk,metadata={"source":"local"}) for chunk in chunks]
print(documents)


# ## Automatic Chunking
text_spllitter = CharacterTextSplitter(chunk_size=40,chunk_overlap=0,separator=' ',strip_whitespace=True)
chunks = text_spllitter.create_documents([text])
print(chunks)

## Recursive Chunking

text_spllitter = RecursiveCharacterTextSplitter(chunk_size=450,chunk_overlap = 0,separators=' ',strip_whitespace=True)
chunks = text_spllitter.create_documents([text])
print(chunks)

## Semantic Chunking
text_splitter = SemanticChunker(OllamaEmbeddings(model="nomic-embed-text"),breakpoint_threshold_type='percentile')
documents = text_splitter.create_documents([text])
print(documents)


