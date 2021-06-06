from numpy import sum,isrealobj,sqrt
from numpy.random import standard_normal

class SignalProcessing():
    @staticmethod
    def awgn(s,SNRdB,L=1):
        """
        AWGN channel
        Add AWGN noise to input signal. The function adds AWGN noise vector to signal 's' to generate a resulting signal vector 'r' of specified SNR in dB. It also
        returns the noise vector 'n' that is added to the signal 's' and the power spectral density N0 of noise added
        Parameters:
            s : input/transmitted signal vector
            SNRdB : desired signal to noise ratio (expressed in dB) for the received signal
            L : oversampling factor (applicable for waveform simulation) default L = 1.
        Returns:
            r : received signal vector (r=s+n)

        author - Mathuranathan Viswanathan (gaussianwaves.com)
        This code is part of the book Digital Modulations using Python

        https://www.gaussianwaves.com/2015/06/how-to-generate-awgn-noise-in-matlaboctave-without-using-in-built-awgn-function/
        """

        gamma = 10**(SNRdB/10) #SNR to linear scale
        
        if s.ndim==1:# if s is single dimensional vector
            P=L*sum(abs(s)**2)/len(s) #Actual power in the vector
        else: # multi-dimensional signals like MFSK
            P=L*sum(sum(abs(s)**2))/len(s) # if s is a matrix [MxN]
        
        N0=P/gamma # Find the noise spectral density
        
        if isrealobj(s):# check if input is real/complex object type
            n = sqrt(N0/2)*standard_normal(s.shape) # computed noise
        else:
            n = sqrt(N0/2)*(standard_normal(s.shape)+1j*standard_normal(s.shape))
        
        r = s + n # received signal

        return r