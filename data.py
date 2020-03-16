import scipy.io
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import normalize
import sklearn.metrics as metrics


TEST_SIZE = 300
TRAIN_SIZE = 400
T = 100

def PCA(Images,mean=None,draw=False,Eigenvec=None,T=T):
    if mean is None:
        mean = np.mean(Images,axis=0)
    A = (Images - mean).swapaxes(0,1)
    if Eigenvec is None:
        B = np.matmul(A.transpose(),A)
        V,W = np.linalg.eigh(B)
        Widx = np.argsort(V)[::-1][:T]
        W = np.matmul(A,W)
        VP = np.take(W,Widx,axis=1)
        #normalize
        V = normalize(VP,axis=0,norm='l2')
        Digits = np.matmul(V.transpose(),Images.swapaxes(0,1))
        print(Digits.shape)

        R = np.matmul(V,Digits).swapaxes(0,1)
        if draw:
            DrawSamples(Images,VP,R)
    else:
        V = Eigenvec
        Digits = np.matmul(V.transpose(),Images.swapaxes(0,1))
    return Digits.swapaxes(0,1), mean, V


def DrawSamples(Images,V,R):
    fig, axs = plt.subplots(4,4)
    for ax in range(len(axs.ravel())):
        axs.flat[ax].imshow(Images[ax].reshape(28,28), cmap='gray', vmin=0,vmax=256)
        axs.flat[ax].axis('off')
    fig.suptitle("Images Samples sample_size="+str(TEST_SIZE)+" num_vec="+str(T))
    plt.savefig("Images Sample_size("+str(TEST_SIZE)+")_T("+str(T)+")",bbox_inches='tight')
    plt.show()

    fig, axs = plt.subplots(4,4)
    for ax in range(len(axs.ravel())):
        axs.flat[ax].imshow(V[:,ax].reshape(28,28), cmap='gray', vmin=0,vmax=256)
        axs.flat[ax].axis('off')
    fig.suptitle("EIgenvec Samples sample_size="+str(TEST_SIZE)+" num_vec="+str(T))
    plt.savefig("Eigenvec Sample_size("+str(TEST_SIZE)+")_T("+str(T)+")",bbox_inches='tight')
    plt.show()

    fig, axs = plt.subplots(4,4)
    for ax in range(len(axs.ravel())):
        axs.flat[ax].imshow(R[ax].reshape(28,28), cmap='gray', vmin=0,vmax=256)
        axs.flat[ax].axis('off')
    fig.suptitle("Reconstruction Samples sample_size="+str(TEST_SIZE)+" num_vec="+str(T))
    plt.savefig("Reconstruction Sample_size("+str(TEST_SIZE)+")_T("+str(T)+")",bbox_inches='tight')
    plt.show()





def showIm(image):
    plt.imshow(image.reshape(28,28),cmap = 'gray',vmin=0,vmax=256)
    plt.show()

def main():
    sounds = scipy.io.loadmat('sounds.mat')['sounds']
    track0 = open("track0","wb")
    track0.write(np.ascontiguousarray(sounds[0]))

    print("lly")




if __name__ == '__main__':
    main()