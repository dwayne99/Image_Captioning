import torch
import torch.nn.functional as F
import numpy as np 
import json
from torchvision import transforms  
import matplotlib.pyplot as plt 
import matplotlib.cm as cm 
import skimage.transform
import argparse
import cv2
from cv2 import imread, resize
from PIL import Image
import warnings
warnings.filterwarnings(action='ignore') 


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def caption_image_beam_search(encoder, decoder, image_path, word_map, beam_size = 3):
    '''
    Reads an image and captions it with beam search

    :param encoder: encoder model
    :param decoder: decoder model
    :param image_path: path to image
    :param word_map: word map
    :param beam_size: number of sequences to consider at each decode-step
    :return: caption, weights for visualization
    '''
    k = beam_size
    vocab_size = len(word_map)

    # Read image and process
    img = imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # convert gray-scale to 3 color channels
    if len(img.shape) == 2:
        img = img[:, :, np.newaxis]
        img = np.concatenate([img, img, img], axis=2)
    # resize the image
    img = resize(img, (256,256))
    # convert (H x W x C) to (C x W x H)
    img = img.transpose(2,0,1)
    # normalize the image in range (0,1)
    img = img / 255.
    # convert img to tensor
    img = torch.FloatTensor(img).to(device)
    # Apply transform to the image tensor
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([normalize])
    image = transform(img)  # (3, 256, 256)

    # Encode 
    image = image.unsqueeze(0) # (1,3,256,256)
    encoder_out = encoder(image) # (1, enc_image_size, enc_image_size, encoder_dim)
    enc_image_size = encoder_out.size(1)
    encoder_dim = encoder_out.size(3)

    # Flattening the encoding
    # (1, enc_image_size * enc_image_size, encoder_dim) = (1, num_pixels, encoder_dim)
    encoder_out = encoder_out.view(1,-1,encoder_dim) 
    num_pixels = encoder_out.size(1)

    # we'll treat the problem as having a batch size of k
    encoder_out = encoder_out.expand(k, num_pixels, encoder_dim) #(k, num_pixels,encoder_dim)

    # Tensor to store top k previous words at each step; now they're just <start>
    k_prev_words = torch.LongTensor([[word_map['<start>']]] * k).to(device)  # (k, 1)

    # Tensor to store top k sequences; now they're just <start>
    seqs = k_prev_words # (k,1)

    # Tensor to store top k sequences' scores; now they're just 0
    top_k_scores = torch.zeros(k,1).to(device) #(k,1)

    # Tensor to store top k sequences' alphas; now they're just 1
    seqs_alpha = torch.ones(k,1, enc_image_size, enc_image_size).to(device)  # (k, 1, enc_image_size, enc_image_size)

    # Lists to store complete sequences, their alphas and scores
    complete_seqs = list()
    complete_seqs_alpha = list()
    complete_seqs_scores = list()

    # start decoding
    step = 1
    h, c = decoder.init_hidden_state(encoder_out)

    # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
    while True:

        embeddings = decoder.embedding(k_prev_words).squeeze(1) # (s,embed_dim)
        
        awe, alpha = decoder.attention(encoder_out, h) # (s, encoder_dim), (s,num_pixels)

        alpha = alpha.view(-1, enc_image_size, enc_image_size)  # (s, enc_image_size, enc_image_size)

        gate = decoder.sigmoid(decoder.f_beta(h))  # gating scalar, (s, encoder_dim)
        awe = gate * awe

        h, c = decoder.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))  # (s, decoder_dim)

        scores = decoder.fc(h) #(s, vocab_size)
        scores = F.log_softmax(scores, dim=1)

        # add
        scores = top_k_scores.expand_as(scores) + scores #(s, vocab_size)

        # for the first step, all k points will have the same scores (since same k previous words, h , c)
        if step ==1:
            top_k_scores, top_k_words = scores[0].topk(k,0,True,True) # (s)
        else:
            # unroll and find top scores, and their unrolled indices
            top_k_scores, top_k_words = scores.view(-1).topk(k,0,True,True) #(s)

        # convert unrolled indices to actual indices of scores
        prev_word_inds = top_k_words / vocab_size # (s)
        next_word_inds = top_k_words % vocab_size # (s)
        
        # add new words to sequences, alphas
        seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)],dim=1) #(s, step + 1)
        seqs_alpha = torch.cat([seqs_alpha[prev_word_inds], alpha[prev_word_inds].unsqueeze(1)],
        dim=1) # (s, step+1, enc_image_size, enc_image_size)

        # which sequences are incomplete (did'nt reach <end>)?
        incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) 
                            if next_word != word_map['<end>']]

        complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

         # Set aside complete sequences
        if len(complete_inds) > 0:
            complete_seqs.extend(seqs[complete_inds].tolist())
            complete_seqs_alpha.extend(seqs_alpha[complete_inds].tolist())
            complete_seqs_scores.extend(top_k_scores[complete_inds])
        k -= len(complete_inds)  # reduce beam length accordingly

        # Proceed with incomplete sequences
        if k == 0:
            break

        seqs = seqs[incomplete_inds]
        seqs_alpha = seqs_alpha[incomplete_inds]
        h = h[prev_word_inds[incomplete_inds]]
        c = c[prev_word_inds[incomplete_inds]]
        encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
        top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
        k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

        # Break if things have been going on too long
        if step > 50:
            break
        step += 1

    i = complete_seqs_scores.index(max(complete_seqs_scores))
    seq = complete_seqs[i]
    alphas = complete_seqs_alpha[i]

    return seq, alphas


# def visualize_att(image_path, seq, alphas, rev_word_map, smooth=True):
#     """
#     Visualizes caption with weights at every word.
#     Adapted from paper authors' repo: https://github.com/kelvinxu/arctic-captions/blob/master/alpha_visualization.ipynb
#     :param image_path: path to image that has been captioned
#     :param seq: caption
#     :param alphas: weights
#     :param rev_word_map: reverse word mapping, i.e. ix2word
#     :param smooth: smooth weights?
#     """
#     image = Image.open(image_path)
#     image = image.resize([14 * 24, 14 * 24], Image.LANCZOS)

#     words = [rev_word_map[ind] for ind in seq]
#     print(' '.join(words[1:-1]))

#     plt.imshow(image)
#     plt.show()
#     plt.savefig('books_read.png')
#     for t in range(len(words)):
#         if t > 50:
#             break
#         plt.subplot(np.ceil(len(words) / 5.), 5, t + 1)

#         plt.text(0, 1, '%s' % (words[t]), color='black', backgroundcolor='white', fontsize=12)
#         current_alpha = alphas[t, :]
#         if smooth:
#             alpha = skimage.transform.pyramid_expand(current_alpha.numpy(), upscale=24, sigma=8)
#         else:
#             alpha = skimage.transform.resize(current_alpha.numpy(), [14 * 24, 14 * 24])
#         if t == 0:
#             plt.imshow(alpha, alpha=0)
#         else:
#             plt.imshow(alpha, alpha=0.8)
#         plt.set_cmap(cm.Greys_r)
#         plt.axis('off')
#     plt.show()
    # plt.savefig('img.jpeg')



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Show, Attend, and Tell - Generate Caption')

    parser.add_argument('--img', '-i', help='path to image')
    parser.add_argument('--model', '-m', help='path to model')
    parser.add_argument('--word_map', '-wm', help='path to word map JSON')
    parser.add_argument('--beam_size', '-b', default=5, type=int, help='beam size for beam search')
    parser.add_argument('--dont_smooth', dest='smooth', action='store_false', help='do not smooth alpha overlay')

    args = parser.parse_args()

    # Load model
    checkpoint = torch.load(args.model, map_location=str(device))
    decoder = checkpoint['decoder']
    decoder = decoder.to(device)
    decoder.eval()
    encoder = checkpoint['encoder']
    encoder = encoder.to(device)
    encoder.eval()

    # Load word map (word2ix)
    with open(args.word_map, 'r') as j:
        word_map = json.load(j)
    rev_word_map = {v: k for k, v in word_map.items()}  # ix2word

    # Encode, decode with attention and beam search
    seq, alphas = caption_image_beam_search(encoder, decoder, args.img, word_map, args.beam_size)
    alphas = torch.FloatTensor(alphas)

    words = [rev_word_map[ind] for ind in seq]
    print(' '.join(words[1:-1]))
    image = Image.open(args.img)
    plt.imshow(image)
    plt.axis('off')
    plt.show()

    # Visualize caption and attention of best sequence
    # visualize_att(args.img, seq, alphas, rev_word_map, args.smooth)
