#!/usr/bin/env python
'''
pyMP3

MP3 decoder

Created on Aug 18, 2010
 
@author: portalfire
'''

#import psyco
#psyco.full()

from bitbfr import BitBfr
from numpy import *
from pyMP3_tables import *
from struct import *
import time
import cProfile

dbg_output = False
frame_cnt_limit = 100

PI = 3.1415

class mp3_frame_hdr:
    def __init__(self):
        self.mpeg_version = 0        
        self.layer = 0
        self.has_CRC = False
        self.bitrate = 0
        self.smpl_rate = 0
        self.padding = False
        self.channel_mode = "unknown"
        self.mode_extention = "unknown"
        self.copyrighted = False
        self.original = False
        self.emphasis = "unknown"
        self.n_channels = 0

class mp3_frame_side_info:
    def __init__(self):
        self.main_data_begin        = 0
        self.private_bits           = 0
        self.scfsi                  = zeros( (2,4), dtype=int32)
        self.part2_3_length         = zeros( (2,2), dtype=int32)
        self.big_values             = zeros( (2,2), dtype=int32)
        self.global_gain            = zeros( (2,2), dtype=int32)
        self.scalefac_compress      = zeros( (2,2), dtype=int32)
        self.window_switching_flag  = zeros( (2,2), dtype=int32)
        self.block_type             = zeros( (2,2), dtype=int32)
        self.mixed_block_flag       = zeros( (2,2), dtype=int32)
        self.table_select           = zeros( (2,2,3), dtype=int32)
        self.subblock_gain          = zeros( (2,2,3), dtype=int32)
        self.region0_count          = zeros( (2,2), dtype=int32)
        self.region1_count          = zeros( (2,2), dtype=int32)
        self.preflag                = zeros( (2,2), dtype=int32)
        self.scalefac_scale         = zeros( (2,2), dtype=int32)
        self.count1table_select     = zeros( (2,2), dtype=int32)

class mp3_frame_main_data:
    def __init__(self):
        self.scalefac_s    = zeros( (2,2,3,13), dtype=int32)
        self.scalefac_l    = zeros( (2,2,23), dtype=int32)
        self.q_vals        = zeros( (2,2,32,18), dtype=int32)

class mp3_frame_decoded_data:
    def __init__(self):
        self.dq_vals          = zeros( (2,2,32,18), dtype=double)
        self.lr_vals          = zeros( (2,2,32,18), dtype=double)
        self.reordered_vals   = zeros( (2,2,32,18), dtype=double)
        self.antialiased_vals = zeros( (2,2,32,18), dtype=double)
        self.hybrid_vals      = zeros( (2,2,32,18), dtype=double)

class mp3_frame:
    def __init__(self):
        self.hdr          = mp3_frame_hdr()
        self.side_info    = mp3_frame_side_info()
        self.main_data    = mp3_frame_main_data()
        self.decoded_data = mp3_frame_decoded_data()

class pyMP3:
    def __init__(self, filename):
        #read entire file into memory
        f = open(filename, "rb")
        file_data = f.read()
        f.close()      
        
        self.bitbfr = BitBfr(file_data)
        
        #info needed block to block        
        self.hybrid_prevblck = zeros( (2,32,18), dtype=double)
        self.sbs_bufOffset   = [64, 64]        
        self.sbs_buf         = zeros( (2,1024), dtype=double)
        self.sbs_filter      = zeros( (64,32), dtype=double)
        self.pcm             = [ [], [] ]

        self.first_frame = True

        self.init_tables()


       
    def init_tables(self):
        #=======MDCT======
        self.mdct_win = zeros( (4,36), dtype=double)
        self.mdct_cos_tbl = zeros( (4*36), dtype=double)
        self.mdct_cos_tbl_2 = zeros( (36,18), dtype=double)
        # block type 0 
        for i in xrange(36):
            self.mdct_win[0][i] = sin( PI/36 *(i+0.5) )

        # block type 1
        for i in xrange(18):
            self.mdct_win[1][i] = sin( PI/36 *(i+0.5) );
        for i in xrange(18, 24):
            self.mdct_win[1][i] = 1.0
        for i in xrange(24, 30):
            self.mdct_win[1][i] = sin( PI/12 *(i+0.5-18) )
        for i in xrange(30, 36):
            self.mdct_win[1][i] = 0.0

        # block type 2 
        for i in xrange(12):
            self.mdct_win[2][i] = sin( PI/12*(i+0.5) ) 
        for i in xrange(12, 36): 
            self.mdct_win[2][i] = 0.0

        #block type 3
        for i in xrange(6):
            self.mdct_win[3][i] = 0.0
        for i in xrange(6, 12):
            self.mdct_win[3][i] = sin( PI/12 *(i+0.5-6) )
        for i in xrange(12, 18):
            self.mdct_win[3][i] =1.0
        for i in xrange(18, 36):
            self.mdct_win[3][i] = sin( PI/36*(i+0.5) )            

        for i in xrange(4*36):
            self.mdct_cos_tbl[i] = cos(PI/(2*36) * i)            

        N = 36
        for p in xrange(N):
            for m in xrange(N/2):
                self.mdct_cos_tbl_2[p][m] = self.mdct_cos_tbl[((2*p+1+N/2)*(2*m+1))%(4*36)]
        
        #======SUB BAND SYNTHESIS===
        
        #create filter
        for i in xrange(64):
            for k in xrange(32):
                self.sbs_filter[i][k] = 1e9*cos(((PI/64*i+PI/4)*(2*k+1)))                 
                if (self.sbs_filter[i][k] >= 0):
                    dummy, self.sbs_filter[i][k] = modf(self.sbs_filter[i][k]+0.5)
                else:
                    dummy, self.sbs_filter[i][k] = modf(self.sbs_filter[i][k]-0.5)
                self.sbs_filter[i][k] *= 1e-9;
            
            
    def decode(self, filename):
        print "Starting Decode"        

        self.frame_num = 0
        while( self.bitbfr.bits_left() > 0):        
            self.cur_frame = mp3_frame()
            st = time.time()
            #Decode the data            
            self.find_next_syncword()
            self.decode_frame_header()
            self.decode_CRC()
            self.decode_side_info()           
            self.decode_main_info()
            
            #Process the Data
            self.dequantize_samples()
            self.process_stereo()
            self.reorder_samples()
            self.antialias_samples()
            self.hybrid_synthesis()
            self.polyphase_synthesis()
            
            #timing            
            tt = (time.time() -st)*1000            
            if(tt == 0): tt=1
            print "frame #%d - %.1f %.3fx" % (self.frame_num+1, tt, 26/tt)
            self.frame_num+=1
            if(self.frame_num == frame_cnt_limit ):
                break

        self.write_wav(filename)

    def write_wav(self, filename):
        file_sz = len(self.pcm[0])*2 + 36
        n_channels = 1
        smpl_rate  = 44100

        f = open(filename, "wb")
        f.write("RIFF")
        f.write( pack("<i", file_sz) )       
        f.write("WAVE")
        f.write("fmt ")
        f.write( pack("<i", 16) )       
        f.write( pack("<h", 1) )       
        f.write( pack("<h", n_channels) ) 
        f.write( pack("<i", smpl_rate) )        
        f.write( pack("<i", smpl_rate*n_channels*2) )       
        f.write( pack("<h", n_channels*2) ) 
        f.write( pack("<h", 16) ) 
        f.write("data")
        f.write( pack("<i", file_sz-36) )        

        for x in self.pcm[0]:
            f.write( pack("<h", x) ) 
            
        f.close()

    def dequantize_samples(self):
        if( dbg_output  and False):
            for gr in xrange(2):
                for sb in xrange(SBLIMIT):
                        for ss in xrange(SSLIMIT):       
                            print "dq", self.frame_num, gr, sb, ss, self.cur_frame.main_data.q_vals[gr][0][sb][ss]


        for gr in xrange(2):        
            for ch in xrange(self.cur_frame.hdr.n_channels):        
                cb = 0                 
                #get initial boundary        
                if (self.cur_frame.side_info.window_switching_flag[ch][gr] and (self.cur_frame.side_info.block_type[ch][gr] == 2) ):
                    if (self.cur_frame.side_info.mixed_block_flag[ch][gr]): 
                        #mixed (long first)                
                        next_cb_boundary = sfBandIndex_l[self.cur_frame.hdr.smpl_rate][1]
                    else: 
                        #pure short                
                        next_cb_boundary = sfBandIndex_s[self.cur_frame.hdr.smpl_rate][1]*3
                        cb_width         = sfBandIndex_s[self.cur_frame.hdr.smpl_rate][1]
                        cb_begin         = 0;
                else: 
                    #long blocks             
                    next_cb_boundary = sfBandIndex_l[self.cur_frame.hdr.smpl_rate][1]

                #apply formula per block type */    
                for sb in xrange(SBLIMIT):
                    for ss in xrange(SSLIMIT): 
                        if ( (sb*18)+ss == next_cb_boundary):  
                            if (self.cur_frame.side_info.window_switching_flag[ch][gr] and (self.cur_frame.side_info.block_type[ch][gr] == 2)): 
                                if (self.cur_frame.side_info.mixed_block_flag[ch][gr]):  
                                    if (((sb*18)+ss) == sfBandIndex_l[self.cur_frame.hdr.smpl_rate][8]):  
                                        next_cb_boundary = sfBandIndex_s[self.cur_frame.hdr.smpl_rate][4]*3 
                                        cb               = 3
                                        cb_width         = sfBandIndex_s[self.cur_frame.hdr.smpl_rate][cb+1] - sfBandIndex_s[self.cur_frame.hdr.smpl_rate][cb]
                                        cb_begin         = sfBandIndex_s[self.cur_frame.hdr.smpl_rate][cb]*3      
                                    elif (((sb*18)+ss) < sfBandIndex_l[self.cur_frame.hdr.smpl_rate][8]):
                                        cb += 1 
                                        next_cb_boundary = sfBandIndex_l[self.cur_frame.hdr.smpl_rate][cb+1]
                                    else: 
                                        cb += 1 
                                        next_cb_boundary = sfBandIndex_s[self.cur_frame.hdr.smpl_rate][cb+1]*3
                                        cb_width         = sfBandIndex_s[self.cur_frame.hdr.smpl_rate][cb+1] - sfBandIndex_s[self.cur_frame.hdr.smpl_rate][cb]
                                        cb_begin         = sfBandIndex_s[self.cur_frame.hdr.smpl_rate][cb]*3                                       
                                else:  
                                    cb += 1
                                    next_cb_boundary = sfBandIndex_s[self.cur_frame.hdr.smpl_rate][cb+1]*3
                                    cb_width = sfBandIndex_s[self.cur_frame.hdr.smpl_rate][cb+1] -sfBandIndex_s[self.cur_frame.hdr.smpl_rate][cb];
                                    cb_begin = sfBandIndex_s[self.cur_frame.hdr.smpl_rate][cb]*3;                                  
                            else: 
                                cb += 1
                                next_cb_boundary = sfBandIndex_l[self.cur_frame.hdr.smpl_rate][cb+1]
                 
                        #Compute global scaling
                        self.cur_frame.decoded_data.dq_vals[gr][ch][sb][ss] = pow( 2.0 , (0.25 * (self.cur_frame.side_info.global_gain[ch][gr] - 210.0)))
                        val_a = self.cur_frame.decoded_data.dq_vals[gr][ch][sb][ss]

                        #Compute block dependent scaling
                        if (self.cur_frame.side_info.window_switching_flag[ch][gr] and (
                            ((self.cur_frame.side_info.block_type[ch][gr] == 2) and (self.cur_frame.side_info.mixed_block_flag[ch][gr] == 0)) or
                            ((self.cur_frame.side_info.block_type[ch][gr] == 2) and self.cur_frame.side_info.mixed_block_flag[ch][gr] and (sb >= 2)) )): 
                            #SHORT
                            self.cur_frame.decoded_data.dq_vals[gr][ch][sb][ss] *= pow(2.0, 0.25 * -8.0 * self.cur_frame.side_info.subblock_gain[ch][gr][(((sb*18)+ss) - cb_begin)/cb_width])
                            val_a2 = self.cur_frame.decoded_data.dq_vals[gr][ch][sb][ss]                            
                            self.cur_frame.decoded_data.dq_vals[gr][ch][sb][ss] *= pow(2.0, 0.25 * -2.0 * 
                                (1.0+self.cur_frame.side_info.scalefac_scale[ch][gr]) * self.cur_frame.main_data.scalefac_s[ch][gr][(((sb*18)+ss) - cb_begin)/cb_width][cb])
                            val_a3 = self.cur_frame.decoded_data.dq_vals[gr][ch][sb][ss]                            
                        else:    
                            #LONG block types 0,1,3 & 1st 2 subbands of switched blocks 
                            self.cur_frame.decoded_data.dq_vals[gr][ch][sb][ss] *= pow(2.0, -0.5 * 
                                (1.0+self.cur_frame.side_info.scalefac_scale[ch][gr]) * (self.cur_frame.main_data.scalefac_l[ch][gr][cb] + self.cur_frame.side_info.preflag[ch][gr] * pretab[cb]))

                        val_b = self.cur_frame.decoded_data.dq_vals[gr][ch][sb][ss]
                        val_b2 = self.cur_frame.side_info.scalefac_scale[ch][gr]
                        val_b3 = self.cur_frame.main_data.scalefac_l[ch][gr][cb]

                        #Scale quantized value        
                        self.cur_frame.decoded_data.dq_vals[gr][ch][sb][ss] *= pow( abs(self.cur_frame.main_data.q_vals[gr][ch][sb][ss]), (4.0/3.0) )
                        if (self.cur_frame.main_data.q_vals[gr][ch][sb][ss]<0):
                            self.cur_frame.decoded_data.dq_vals[gr][ch][sb][ss] = -self.cur_frame.decoded_data.dq_vals[gr][ch][sb][ss]

                        val_c = self.cur_frame.decoded_data.dq_vals[gr][ch][sb][ss]

    def process_stereo(self):
        has_intensity_stereo = (self.cur_frame.hdr.mode_extention & INTENSITY_STEREO_BIT)
        has_mid_side_stereo = (self.cur_frame.hdr.mode_extention & MID_SIDE_STEREO_BIT)
        lr      = self.cur_frame.decoded_data.lr_vals
        dq_vals = self.cur_frame.decoded_data.dq_vals
        

        if( dbg_output and False ):
            for gr in xrange(2):
                for sb in xrange(SBLIMIT):
                        for ss in xrange(SSLIMIT):       
                            print "s", self.frame_num, gr, sb, ss, dq_vals[gr][0][sb][ss]

        for gr in xrange(2):                       
            if (self.cur_frame.hdr.n_channels==2): 
                is_pos = [7 for dummy in xrange(576)]

                if(has_intensity_stereo):
                    is_ratio = process_intensity_stereo(self, gr, is_pos)
                            
                for sb in xrange(SBLIMIT):
                    for ss in xrange(SSLIMIT): 
                        i = (sb*18)+ss
                        if ( is_pos[i] == 7 ): 
                            if (has_mid_side_stereo ): 
                                lr[gr][0][sb][ss] = (dq_vals[gr][0][sb][ss]+dq_vals[gr][1][sb][ss])/1.41421356
                                lr[gr][1][sb][ss] = (dq_vals[gr][0][sb][ss]-dq_vals[gr][1][sb][ss])/1.41421356                   
                            else: 
                                lr[gr][0][sb][ss] = dq_vals[gr][0][sb][ss]
                                lr[gr][1][sb][ss] = dq_vals[gr][1][sb][ss]
                        elif (has_intensity_stereo ): 
                            lr[gr][0][sb][ss] = dq_vals[gr][0][sb][ss] * (is_ratio[i]/(1+is_ratio[i]))
                            lr[gr][1][sb][ss] = dq_vals[gr][0][sb][ss] * (1/(1+is_ratio[i])) 
                        
                        else: 
                            print "Stereo - Error"                                                   
                     
            else:                 
                for sb in xrange(SBLIMIT):
                    for ss in xrange(SSLIMIT): 
                        lr[gr][0][sb][ss] = dq_vals[gr][0][sb][ss]


    def process_intensity_stereo(self, gr, is_pos):
        ch = 0  

        if (self.cur_frame.side_info.window_switching_flag[ch][gr] and (self.cur_frame.side_info.block_type[ch][gr] == 2)):
            is_ratio = process_intensity_stereo__non_standard_window(gr, is_pos)        
        else:
            is_ratio = process_intensity_stereo__standard_window(gr, is_pos)

        return is_ratio

    def process_intensity_stereo__standard_window(self, gr, is_pos):
        smpl_rate = self.cur_frame.hdr.smpl_rate        
        i  = 31
        ss = 17
        sb = 0;
        while ( i >= 0 ):
            if ( self.decoded_data.dq_vals[1][i][ss] != 0.0 ):
                sb = i*18+ss
                i  = -1
            else:
                ss-=1
                if ( ss < 0 ):
                    i -= 1
                    ss = 17
        i = 0
        while ( sfBandIndex_l[smpl_rate][i] <= sb ):
            i += 1

        sfb = i
        i = sfBandIndex_l[smpl_rate][i]
        for sfb in xrange(sfb,21):
            sb = sfBandIndex_l[smpl_rate][sfb+1] - sfBandIndex_l[smpl_rate][sfb]
            for sb in xrange(sb, 0, -1):
                is_pos[i] =  self.cur_frame.main_data.scalefac_l[1][gr][sfb]
                if ( is_pos[i] != 7 ):
                    is_ratio[i] = tan( is_pos[i] * (PI / 12))
                i+=1

        sfb = sfBandIndex_l[smpl_rate][20]
        for sb in xrange(576 - sfBandIndex_l[smpl_rate][21],0,-1):
            is_pos[i]   = is_pos[sfb]
            is_ratio[i] = is_ratio[sfb]
            i += 1
        
        return is_ratio

    def process_intensity_stereo__non_standard_window(self, gr, is_pos):
        ch = 0  
        smpl_rate = self.cur_frame.hdr.smpl_rate

        if (self.cur_frame.side_info.mixed_block_flag[ch][gr]):
            max_sfb = 0;

            for j in xrange(3):
                sfbcnt = 2
                for sfb in xrange(12,2,-1):                   
                    lines = sfBandIndex_s[smpl_rate][sfb+1]-sfBandIndex_s[smpl_rate][sfb];
                    i = 3*sfBandIndex_s[sfreq][sfb] + (j+1) * lines - 1;
                    while ( lines > 0 ):
                        if ( self.decoded_data.dq_vals[1][i/SSLIMIT][i%SSLIMIT] != 0.0 ):
                            sfbcnt = sfb
                            sfb = -10
                            lines = -10
                        lines-=1
                        i-=1

                sfb = sfbcnt + 1

                if ( sfb > max_sfb ):
                    max_sfb = sfb

                while( sfb<12 ):
                    sb = sfBandIndex_s[smpl_rate][sfb+1]-sfBandIndex_s[sfreq][sfb]
                    i = 3*sfBandIndex_s[smpl_rate][sfb] + j * sb
                    for sb in xrange(sb,0,-1):
                        is_pos[i] = self.cur_frame.main_data.scalefac_s[1][gr][j][sfb]
                        if ( is_pos[i] != 7 ):
                            is_ratio[i] = tan( is_pos[i] * (PI / 12))
                        i += 1
                    sfb += 1

                sb = sfBandIndex_s[smpl_rate][11]-sfBandIndex_s[smpl_rate][10];
                sfb = 3*sfBandIndex_s[smpl_rate].s[10] + j * sb;
                sb = sfBandIndex_s[smpl_rate][12]-sfBandIndex_s[smpl_rate][11];
                i = 3*sfBandIndex_s[smpl_rate][11] + j * sb;
                for sb in xrange(sb,0,-1):
                    is_pos[i]   = is_pos[sfb]
                    is_ratio[i] = is_ratio[sfb]
                    i += 1

            if ( max_sfb <= 3 ):
                i = 2
                ss = 17
                sb = -1
                while ( i >= 0 ):
                    if ( self.decoded_data.dq_vals[1][i][ss] != 0.0 ):
                        sb = i*18+ss
                        i = -1
                    else:
                        ss-=1
                        if ( ss < 0 ):
                            i-=1
                            ss = 17
                i = 0
                while ( sfBandIndex_l[smpl_rate][i] <= sb ):
                    i+=1
                sfb = i
                i = sfBandIndex_l[smpl_rate][i]
                for sfb in xrange(sfb, 8):
                    sb = sfBandIndex_l[smpl_rate][sfb+1]-sfBandIndex_l[sfreq][sfb]
                    for sb in xrange(sb, 0, -1):
                        is_pos[i] = self.cur_frame.main_data.scalefac_l[1][gr][sfb];
                        if ( is_pos[i] != 7 ):
                            is_ratio[i] = tan( is_pos[i] * (PI / 12))
                        i+=1

        else:
            for j in xrange(3):
                sfbcnt = -1
                for sfb in xrange(12,-1,-1):
                    lines = sfBandIndex_s[smpl_rate][sfb+1]-sfBandIndex_s[smpl_rate][sfb]
                    i = 3*sfBandIndex_s[smpl_rate][sfb] + (j+1) * lines - 1
                    while ( lines > 0 ):
                        if ( self.decoded_data.dq_vals[1][i/SSLIMIT][i%SSLIMIT] != 0.0 ):
                            sfbcnt = sfb
                            sfb = -10
                            lines = -10                         
                        lines-=1
                        i-=1
            sfb = sfbcnt + 1
            while( sfb<12 ):
                sb = sfBandIndex_s[smpl_rate][sfb+1]-sfBandIndex_s[smpl_rate][sfb]
                i = 3*sfBandIndex_s[smpl_rate][sfb] + j * sb
                for sb in xrange(sb,0,-1):
                    is_pos[i] = self.cur_frame.main_data.scalefac_s[1][gr][j][sfb]
                    if ( is_pos[i] != 7 ):
                        is_ratio[i] = tan( is_pos[i] * (PI / 12))
                    i+=1
                sfb+=1

            sb = sfBandIndex_s[smpl_rate][11]-sfBandIndex_s[sfreq][10]
            sfb = 3*sfBandIndex_s[smpl_rate][10] + j * sb
            sb = sfBandIndex_s[smpl_rate][12]-sfBandIndex_s[sfreq][11]
            i = 3*sfBandIndex_s[smpl_rate][11] + j * sb
            for sb in xrange(sb,-1,-1):
                is_pos[i]   = is_pos[sfb]
                is_ratio[i] = is_ratio[sfb]
                i+=1

        return is_ratio
    
    def reorder_samples(self):
        smpl_rate = self.cur_frame.hdr.smpl_rate
        lr   = self.cur_frame.decoded_data.lr_vals
        ro   = self.cur_frame.decoded_data.reordered_vals

        if( dbg_output  and False):
            for gr in xrange(2):
                for sb in xrange(SBLIMIT):
                        for ss in xrange(SSLIMIT):       
                            print "ro", self.frame_num, gr, sb, ss, lr[gr][0][sb][ss]

        
        for gr in xrange(2):
            for ch in xrange(self.cur_frame.hdr.n_channels):
                if (self.cur_frame.side_info.window_switching_flag[ch][gr] and (self.cur_frame.side_info.block_type[ch][gr] == 2)):
                    if (self.cur_frame.side_info.mixed_block_flag[ch][gr]):
                        # NO REORDER FOR LOW 2 SUBBANDS 
                        for sb in xrange(2):
                            for ss in xrange(SSLIMIT):
                                ro[gr][ch][sb][ss] = lr[gr][ch][sb][ss];
                    
                        # REORDERING FOR REST SWITCHED SHORT 
                        sfb_start=sfBandIndex_s[smpl_rate][3]
                        sfb_lines=sfBandIndex_s[smpl_rate][4] - sfb_start
             
                        for sfb in xrange(3,13):
                            for window in xrange(3):                      
                                for freq in xrange(sfb_lines):
                                    src_line = sfb_start*3 + window*sfb_lines + freq 
                                    des_line = (sfb_start*3) + window + (freq*3)
                                    ro[gr][ch][des_line/SSLIMIT][des_line%SSLIMIT] = lr[gr][ch][src_line/SSLIMIT][src_line%SSLIMIT]

                            sfb_start=sfBandIndex_s[smpl_rate][sfb]
                            sfb_lines=sfBandIndex_s[smpl_rate][sfb+1] - sfb_start
                               
                       
                    else: 
                        #pure short                
                        sfb_start=0
                        sfb_lines=sfBandIndex_s[smpl_rate][1] 
                        for sfb in xrange(13): 
                            for window in xrange(3):
                                for freq in xrange(sfb_lines):
                                    src_line = sfb_start*3 + window*sfb_lines + freq 
                                    des_line = (sfb_start*3) + window + (freq*3)
                                    ro[gr][ch][des_line/SSLIMIT][des_line%SSLIMIT] = lr[gr][ch][src_line/SSLIMIT][src_line%SSLIMIT]

                            sfb_start=sfBandIndex_s[smpl_rate][sfb]
                            sfb_lines=sfBandIndex_s[smpl_rate][sfb+1] - sfb_start

                else:
                    #long blocks */
                    for sb in xrange(SBLIMIT):
                        for ss in xrange(SSLIMIT): 
                            ro[gr][ch][sb][ss] = lr[gr][ch][sb][ss]
               

    def antialias_samples(self):
        cs = [ 1.0/sqrt(1.0 + ci*ci) for ci in antialias_Ci]
        ca = [  ci/sqrt(1.0 + ci*ci) for ci in antialias_Ci]

        ro   = self.cur_frame.decoded_data.reordered_vals
        aa   = self.cur_frame.decoded_data.antialiased_vals               

        if( dbg_output and False):
            for gr in xrange(2):
                for sb in xrange(SBLIMIT):
                        for ss in xrange(SSLIMIT):       
                            print "aa", self.frame_num, gr, sb, ss, ro[gr][0][sb][ss]

        for gr in xrange(2):
            for ch in xrange(self.cur_frame.hdr.n_channels):          
                for sb in xrange(SBLIMIT):
                    for ss in xrange(SSLIMIT): 
                        aa[gr][ch][sb][ss] = ro[gr][ch][sb][ss]

                if  ((self.cur_frame.side_info.window_switching_flag[ch][gr] and (self.cur_frame.side_info.block_type[ch][gr] == 2)) and 
                    (not self.cur_frame.side_info.mixed_block_flag[ch][gr])):                                        
                    continue

                if ( self.cur_frame.side_info.window_switching_flag[ch][gr] and self.cur_frame.side_info.mixed_block_flag[ch][gr] and
                 (self.cur_frame.side_info.block_type[ch][gr] == 2)):
                    sblim = 1
                else:
                    sblim = SBLIMIT-1

                # 31 alias-reduction operations between each pair of sub-bands 
                # with 8 butterflies between each pair                         
                for sb in xrange(sblim):   
                    for ss in xrange(8):       
                        bu = ro[gr][ch][sb][17-ss];
                        bd = ro[gr][ch][sb+1][ss];
                        aa[gr][ch][sb][17-ss] = (bu * cs[ss]) - (bd * ca[ss])
                        aa[gr][ch][sb+1][ss]  = (bd * cs[ss]) + (bu * ca[ss])
                       

    def hybrid_synthesis(self):
        data_in  = self.cur_frame.decoded_data.antialiased_vals
        data_out = self.cur_frame.decoded_data.hybrid_vals
        
        if( dbg_output and False ):
            for gr in xrange(2):
                for sb in xrange(SBLIMIT):
                        for ss in xrange(SSLIMIT):       
                            print "hs", self.frame_num, gr, sb, ss, self.cur_frame.decoded_data.antialiased_vals[gr][0][sb][ss]
        
        for gr in xrange(2):
            for ch in xrange(self.cur_frame.hdr.n_channels):        
                for sb in xrange(SBLIMIT):
                    if(self.cur_frame.side_info.window_switching_flag[ch][gr] and self.cur_frame.side_info.mixed_block_flag[ch][gr] and sb<2):
                        blk_type = 0
                    else:
                        blk_type = self.cur_frame.side_info.block_type[ch][gr]

                    rawout = self.inv_mdct( data_in[gr][ch][sb], blk_type)

                    #overlap addition
                    for ss in xrange(SSLIMIT):       
                        data_out[gr][ch][sb][ss] = rawout[ss] + self.hybrid_prevblck[ch][sb][ss]
                        self.hybrid_prevblck[ch][sb][ss] = rawout[ss+18]

        
    def inv_mdct(self, data_in, blk_type):
        out = zeros( (36), dtype=double)

        if(blk_type == 2):
            tmp = zeros( (12), dtype=double)
            N=12
            for i in xrange(3):
                for p in xrange(N):             
                    total = 0.0
                    for m in xrange(N/2):
                        total += data_in[i+3*m] * cos( PI/(2*N)*(2*p+1+N/2)*(2*m+1) )
                    tmp[p] = total * self.mdct_win[blk_type][p]               
                for p in xrange(N):
                    out[6*i+p+6] += tmp[p]
        else:
            cos_tbl = self.mdct_cos_tbl_2
            win     = self.mdct_win[blk_type]
            for p in xrange(36):
                out[p] = sum(data_in * cos_tbl[p]) * win[p] 

        return out                

    def polyphase_synthesis(self):
        data_in  = self.cur_frame.decoded_data.hybrid_vals  
        data_out = self.pcm
        bfr = zeros( (SBLIMIT), dtype=double)

        if( dbg_output and False):
            for gr in xrange(2):
                for sb in xrange(SBLIMIT):
                        for ss in xrange(SSLIMIT):       
                            print "pps %d %d %d %d %.3e" % ( self.frame_num, gr, sb, ss, self.cur_frame.decoded_data.hybrid_vals[gr][0][sb][ss])

        for gr in xrange(2):
            for ch in xrange(self.cur_frame.hdr.n_channels):        
                #frequency inversion
                for ss in xrange(18): 
                    for sb in xrange(SBLIMIT):
                        if ((ss%2) and (sb%2)):
                            self.cur_frame.decoded_data.hybrid_vals[gr][ch][sb][ss] = -self.cur_frame.decoded_data.hybrid_vals[gr][ch][sb][ss]        
                    
                #subBand synthesis
                for ss in xrange(18):
                    for sb in xrange(SBLIMIT):
                        bfr[sb] = data_in[gr][ch][sb][ss]
                    pcm_bfr =self.subBandSynthesis(bfr, ch)

                    #copy to final output
                    data_out[ch].extend(pcm_bfr)

    def subBandSynthesis(self, data_in, ch):
        pcm = zeros( (32), dtype=int32)        

        self.sbs_buf[ch] = roll(self.sbs_buf[ch], 64)

        #init
        bfr = self.sbs_buf[ch]
        
        #fir filter
        for i in xrange(64):
            bfr[i] = sum(data_in * self.sbs_filter[i])            
          
        #window output
        for j in xrange(32):
            total = 0
            for i in xrange(16):
                k = j + i*32
                idx = (k + ( ((i+1)>>1) * 64) )
                total += sbs_window[k] * bfr[idx]                              

            #clip
            pcm[j] = min(max(total * SCALE, -SCALE), SCALE-1)            
        return pcm                    

    def decode_main_info(self):
        for gr in xrange(2):
            for ch in xrange(self.cur_frame.hdr.n_channels):
                #Get scale factors        
                part2_start = self.bitbfr.get_pos()                
                if(self.cur_frame.side_info.window_switching_flag[ch][gr] == 1 and 
                   self.cur_frame.side_info.block_type[ch][gr] == 2):
                    if(self.cur_frame.side_info.mixed_block_flag[ch][gr]):
                        #mixed blocks
                        print "mixed scale blocks not supported yet"
                    else:
                        #short blocks
                        for i in xrange(2):
                            for sfb in xrange(sfbtable_s[i], sfbtable_s[i+1]):
                                for window in xrange(3):
                                    self.cur_frame.main_data.scalefac_s[ch][gr][window][sfb] = self.bitbfr.read_bits(slen[i][self.cur_frame.side_info.scalefac_compress[ch][gr]])
                                                
                        sfb = 12                        
                        for window in xrange(3):
                            self.cur_frame.main_data.scalefac_s[ch][gr][window][sfb] = 0;  
  
                else:
                    #long blocks                   
                    for i in xrange(4):
                        if ((self.cur_frame.side_info.scfsi[ch][i] == 0) or (gr == 0)):
                            for sfb in xrange( sfbtable_l[i], sfbtable_l[i+1]):
                                if(i<2): 
                                    k=0
                                else:    
                                     k=1
                                self.cur_frame.main_data.scalefac_l[ch][gr][sfb] = self.bitbfr.read_bits(slen[k][self.cur_frame.side_info.scalefac_compress[ch][gr]])
                    self.cur_frame.main_data.scalefac_l[ch][gr][22] = 0

                self.decode_main_data_huffman(ch, gr, part2_start)

    def decode_main_data_huffman(self, ch, gr, part2_start):
        
        #calculate region boundries
        if(self.cur_frame.side_info.window_switching_flag[ch][gr] == 1 and 
           self.cur_frame.side_info.block_type[ch][gr] == 2):
            #short block
            region1Start = 36; 
            region2Start = 576; 
        else:
            #long block
            region1Start = sfBandIndex_l[self.cur_frame.hdr.smpl_rate][self.cur_frame.side_info.region0_count[ch][gr] + 1]
            region2Start = sfBandIndex_l[self.cur_frame.hdr.smpl_rate][self.cur_frame.side_info.region0_count[ch][gr] + self.cur_frame.side_info.region1_count[ch][gr] + 2]

        #read big value area
        for i in xrange(0, self.cur_frame.side_info.big_values[ch][gr]*2, 2):
            if (i<region1Start): 
                ht_idx = self.cur_frame.side_info.table_select[ch][gr][0]
            elif (i<region2Start): 
                ht_idx = self.cur_frame.side_info.table_select[ch][gr][1]
            else:               
                ht_idx = self.cur_frame.side_info.table_select[ch][gr][2]
            ht = ht_list[ht_idx]
            v, w, x, y = self.huffman_decoder(ht)
            self.cur_frame.main_data.q_vals[gr][ch][i/SSLIMIT][i%SSLIMIT]           = x
            self.cur_frame.main_data.q_vals[gr][ch][(i+1)/SSLIMIT][(i+1)%SSLIMIT]   = y

        #read count1 area
        idx = self.cur_frame.side_info.big_values[ch][gr]*2        
        ht = ht_list[self.cur_frame.side_info.count1table_select[ch][gr] + 32] #32 is offset to count1 tables
        while( (self.bitbfr.get_pos() < (part2_start + self.cur_frame.side_info.part2_3_length[ch][gr]) ) and (idx<SBLIMIT*SSLIMIT) ):
            v, w, x, y = self.huffman_decoder(ht)
            self.cur_frame.main_data.q_vals[gr][ch][idx/SSLIMIT][idx%SSLIMIT] = v;
            self.cur_frame.main_data.q_vals[gr][ch][(idx+1)/SSLIMIT][(idx+1)%SSLIMIT] = w
            self.cur_frame.main_data.q_vals[gr][ch][(idx+2)/SSLIMIT][(idx+2)%SSLIMIT] = x
            self.cur_frame.main_data.q_vals[gr][ch][(idx+3)/SSLIMIT][(idx+3)%SSLIMIT] = y
            idx += 4

        #the rest are zeros 
        

    def huffman_decoder(self, ht):                
        MXOFF = 250         
        v = w = x = y = 0        
        
        #check for empty tree        
        if(ht.treelen == 0):
            return (0,0,0,0)

        #run through huffman tree
        success = False
        pos = 0
        nbits = 0       
        while(pos<ht.treelen and nbits<32):
            #check for end of tree   
            if(ht.values[pos][0] == 0):         
                x = ht.values[pos][1] >> 4;
                y = ht.values[pos][1] & 0xf;
                success = True
                break

            #get more bits to transverse tree            
            bit = self.bitbfr.read_bits(1)
            while(ht.values[pos][bit] >= MXOFF):
                pos += ht.values[pos][bit]
            pos += ht.values[pos][bit]

            nbits += 1

        if(not success):
            print "Failure during huffman decode"
            return (0,0,0,0)
            
        #read sign bits
        if(ht.tbl_type == "quad"):
            v = (y>>3) & 1;
            w = (y>>2) & 1;
            x = (y>>1) & 1;
            y = y & 1; 

            if(v != 0):
                if(self.bitbfr.read_bits(1)):
                    v = -v                   
            if(w != 0):
                if(self.bitbfr.read_bits(1)):
                    w = -w                   
            if(x != 0):
                if(self.bitbfr.read_bits(1)):
                    x = -x                   
            if(y != 0):
                if(self.bitbfr.read_bits(1)):
                    y = -y                   
        else:
            #process escape encodings            
            if(ht.linbits > 0):
                if((ht.xlen-1) == x):
                    x += self.bitbfr.read_bits(ht.linbits)
            if(x != 0):
                if(self.bitbfr.read_bits(1)):
                    x = -x                   

            if(ht.linbits > 0):
                if((ht.ylen-1) == y):
                    y += self.bitbfr.read_bits(ht.linbits)
            if(y != 0):
                if(self.bitbfr.read_bits(1)):
                    y = -y                   
                
        return (v, w, x, y)        

    def decode_side_info(self):
        self.cur_frame.side_info.main_data_begin = self.bitbfr.read_bits(9)

        if(self.cur_frame.side_info.main_data_begin != 0):
            print "Error - dont support bitreservoir yet"

        if(self.cur_frame.hdr.n_channels==1):
            self.cur_frame.side_info.private_bits = self.bitbfr.read_bits(5)
        else:
            self.cur_frame.side_info.private_bits = self.bitbfr.read_bits(3)

        for ch in xrange(self.cur_frame.hdr.n_channels):
            for i in xrange(4):
                self.cur_frame.side_info.scfsi[ch][i] = self.bitbfr.read_bits(1)

        for gr in xrange(2):
             for ch in xrange(self.cur_frame.hdr.n_channels):
                  self.cur_frame.side_info.part2_3_length[ch][gr] = self.bitbfr.read_bits(12)
                  self.cur_frame.side_info.big_values[ch][gr] = self.bitbfr.read_bits(9)
                  self.cur_frame.side_info.global_gain[ch][gr] = self.bitbfr.read_bits(8)
                  self.cur_frame.side_info.scalefac_compress[ch][gr] = self.bitbfr.read_bits(4)
                  self.cur_frame.side_info.window_switching_flag[ch][gr] = self.bitbfr.read_bits(1)
                  if(self.cur_frame.side_info.window_switching_flag[ch][gr]):
                      self.cur_frame.side_info.block_type[ch][gr] = self.bitbfr.read_bits(2)
                      self.cur_frame.side_info.mixed_block_flag[ch][gr] = self.bitbfr.read_bits(1)
                      for i in xrange(2):
                           self.cur_frame.side_info.table_select[ch][gr][i] = self.bitbfr.read_bits(5)
                      for i in xrange(3):
                           self.cur_frame.side_info.subblock_gain[ch][gr][i] = self.bitbfr.read_bits(3)
                      if(self.cur_frame.side_info.block_type[ch][gr] == 2 and self.cur_frame.side_info.mixed_block_flag[ch][gr] == 0):
                          self.cur_frame.side_info.region0_count[ch][gr] = 8
                      else:
                          self.cur_frame.side_info.region0_count[ch][gr] = 7
                      self.cur_frame.side_info.region1_count[ch][gr] = 20 - self.cur_frame.side_info.region0_count[ch][gr]
                  else:
                       for i in xrange(3):
                           self.cur_frame.side_info.table_select[ch][gr][i] = self.bitbfr.read_bits(5)
                       self.cur_frame.side_info.region0_count[ch][gr] = self.bitbfr.read_bits(4)
                       self.cur_frame.side_info.region1_count[ch][gr] = self.bitbfr.read_bits(3)
                       self.cur_frame.side_info.block_type[ch][gr] = 0
                  self.cur_frame.side_info.preflag[ch][gr] = self.bitbfr.read_bits(1)
                  self.cur_frame.side_info.scalefac_scale[ch][gr] = self.bitbfr.read_bits(1)
                  self.cur_frame.side_info.count1table_select[ch][gr] = self.bitbfr.read_bits(1)

        if(dbg_output):
            print "main_data_begin:",  self.cur_frame.side_info.main_data_begin     
            print "privatebits:", self.cur_frame.side_info.private_bits         
            print "scale factor select info:", self.cur_frame.side_info.scfsi                 
            print "part 2_3 len:", self.cur_frame.side_info.part2_3_length        
            print "big values:", self.cur_frame.side_info.big_values          
            print "global gain:", self.cur_frame.side_info.global_gain           
            print "scalefac_compress", self.cur_frame.side_info.scalefac_compress     
            print "win switch flag:", self.cur_frame.side_info.window_switching_flag 
            print "block type:", self.cur_frame.side_info.block_type            
            print "mixed block flag:", self.cur_frame.side_info.mixed_block_flag      
            print "table select:", self.cur_frame.side_info.table_select          
            print "subblock gain:", self.cur_frame.side_info.subblock_gain         
            print "region0_count", self.cur_frame.side_info.region0_count         
            print "region1_count", self.cur_frame.side_info.region1_count         
            print "preflag", self.cur_frame.side_info.preflag               
            print "scalefac_scale", self.cur_frame.side_info.scalefac_scale        
            print "count1table select:", self.cur_frame.side_info.count1table_select    

    def decode_CRC(self):
         if(self.cur_frame.hdr.has_CRC):
            crc = self.bitbfr.read_bits(16)

    def decode_frame_header(self):
  
        self.cur_frame.hdr.mpeg_version   = mp3_hdr_ver_tbl[self.bitbfr.read_bits(1)]  
        self.cur_frame.hdr.layer          = mp3_hdr_layer_tbl[self.bitbfr.read_bits(2)]
        self.cur_frame.hdr.has_CRC        = not self.bitbfr.read_bits(1)
        self.cur_frame.hdr.bitrate        = mp3_hdr_bitrate_tbl[self.bitbfr.read_bits(4)]
        self.cur_frame.hdr.smpl_rate      = mp3_hdr_smpl_rate_tbl[self.bitbfr.read_bits(2)]
        self.cur_frame.hdr.padding        = self.bitbfr.read_bits(1)
        self.bitbfr.read_bits(1)          #private bit
        self.cur_frame.hdr.channel_mode   = mp3_hdr_channel_mode_tbl[self.bitbfr.read_bits(2)]
        self.cur_frame.hdr.mode_extention = self.bitbfr.read_bits(2)
        self.cur_frame.hdr.copyrighted    = self.bitbfr.read_bits(1)
        self.cur_frame.hdr.original       = self.bitbfr.read_bits(1)  
        self.cur_frame.hdr.emphasis       = mp3_hdr_emphasis_tbl[self.bitbfr.read_bits(2)]  

        if(self.cur_frame.hdr.channel_mode == "mono"):         
            self.cur_frame.hdr.n_channels = 1
        else:
            self.cur_frame.hdr.n_channels = 2

        if(dbg_output):
            print "frame header info"            
            print "ver:", self.cur_frame.hdr.mpeg_version
            print "layer:", self.cur_frame.hdr.layer
            print "has CRC:", self.cur_frame.hdr.has_CRC
            print "bitrate(kps):", self.cur_frame.hdr.bitrate
            print "sample rate:", self.cur_frame.hdr.smpl_rate      
            print "padding:", self.cur_frame.hdr.padding        
            print "channel mode:", self.cur_frame.hdr.channel_mode   
            print "n channels:", self.cur_frame.hdr.n_channels            
            print "mode extention:", self.cur_frame.hdr.mode_extention 
            print "copyrighted:", self.cur_frame.hdr.copyrighted              
            print "original copy:", self.cur_frame.hdr.original                
            print "emphasis:", self.cur_frame.hdr.emphasis               

        if(self.first_frame):
            print "bitrate(kps):", self.cur_frame.hdr.bitrate
            print "sample rate:", self.cur_frame.hdr.smpl_rate      
            print "channel mode:", self.cur_frame.hdr.channel_mode   
            self.first_frame = False

    def find_next_syncword(self):
        #align to byte boundry
        align = self.bitbfr.get_pos()%8
        if(align != 0):        
            self.bitbfr.read_bits(8-align)
    
        cnt = 0        
        while( self.bitbfr.bits_left() > 0):
            b = self.bitbfr.read_bits(4)
            if(b == 0xf):  
                cnt += 1
                if(cnt == 3):
                    break
            else:                
                cnt = 0

        if(dbg_output):        
            print "sync found at %d" % (self.bitbfr.get_pos())


if __name__ == '__main__':
    #test code
    profile_code = False

    print " pyMP3 v.01 "
    print "============"    
    
    src_filename = raw_input("Input filename[test.mp3]:")
    if(src_filename == ""): 
        src_filename = "test.mp3"

    dst_filename = raw_input("Output filename[test.wav]:")
    if(dst_filename == ""): 
        dst_filename = "test.wav"

    frame_cnt_limit = raw_input("Limit number of frames decoded(0=no_limit)[100]:")
    if(frame_cnt_limit == ""): 
        frame_cnt_limit = 100


    print "Loading MP3 into memory"    
    p = pyMP3(src_filename)

    if(profile_code):
        cProfile.run("p.decode(dst_filename)", 'stats.dat')
        import pstats
        ps = pstats.Stats('stats.dat')
        ps.sort_stats('cumulative').print_stats(10)
        ps.sort_stats('time').print_stats(10)
    else:    
        p.decode(dst_filename)
