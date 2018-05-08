class BitBfr:
    def __init__(self, s = []):
        self.s=[ord(x) for x in s]
        self.pos = 0
        self.bit_len = len(s)*8
        
    def init_with_int_array(self, s):
        self.s=s
        self.pos = 0
        self.bit_len = len(s)*8
        
    def seek_abs(self, pos):
        self.pos = pos

    def seek_rel(self, ofst):
        self.pos += ofst

    def read_bits(self, n_bits):
        val = 0
        bits_read = 0
        
        while(bits_read < n_bits):
            val = val << 1
            src_byte = self.s[ self.pos//8 ]
            src_bit = src_byte & (0x80>>(self.pos%8))
            if(src_bit != 0):
                val |= 1
            self.pos  += 1
            bits_read += 1
        #print "rb %d %d" % (n_bits, val)        
        return val      
    
    def bits_left(self):
        if(self.bit_len <= self.pos):
            return 0
        else:
            return (self.bit_len - self.pos)

    def get_pos(self):
        return self.pos

