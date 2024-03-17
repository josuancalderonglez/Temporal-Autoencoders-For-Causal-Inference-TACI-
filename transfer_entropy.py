
import jpype as jp
from jpype import startJVM, getDefaultJVMPath, JArray, JInt, JDouble, JPackage

class transfer_entropy:
     
    def te(sourceArray, destArray , lag, print_out=False):
        
        # Create a Kraskov TE calculator:
        teCalcClass = JPackage("infodynamics.measures.continuous.kraskov").TransferEntropyCalculatorKraskov
        teCalc = teCalcClass()

        teCalc.setProperty("NORMALISE", "true") # Normalise the individual variables
        teCalc.initialise(1) # Use history length 1 (Schreiber k=1)
        teCalc.setProperty("k", "4") # Use Kraskov parameter K=4 for 4 nearest points
        teCalc.setProperty("kNN", "8")
        teCalc.setProperty("sourceLag", str(lag))

        teCalc.setObservations(JArray(JDouble, 1)(sourceArray), JArray(JDouble, 1)(destArray))
        result = teCalc.computeAverageLocalOfObservations()

        if print_out:
            print(("TE(variable_x->variable_y) was %.3f nats" % (result)))

        return result

    def te_pairwise(directory, variable_x,variable_y, lag):

        try:
            jarLocation = directory + "../infodynamics/infodynamics.jar"
            startJVM(getDefaultJVMPath(), "-ea", "-Djava.class.path=" + jarLocation)
        except:
            pass #Exception usually raised when JVM is already running

        score_y = transfer_entropy.te(variable_x, variable_y, lag, print_out=False)
        score_x = transfer_entropy.te(variable_y, variable_x, lag, print_out=False)

        # jp.shutdownJVM()

        return score_x, score_y