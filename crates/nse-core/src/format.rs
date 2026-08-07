//! The `.nse` binary file format.
//!
//! On-disk layout (all integers little-endian):
//!
//! ```text
//! offset  section
//! 0       NSEFileHeader        (fixed, packed)
//! ...     dense core           (outlier weights, FP16/INT8 — size = header.dense_core_size)
//! ...     codebook             (shared PQ codebook — size = header.codebook_size)
//! ...     micro-expert meta    (array of MicroExpertMeta + centroids)
//! ...     micro-expert data    (packed ternary / PQ indices)
//! EOF     MIPS index tree      (starts at header.index_tree_offset)
//! ```
//!
//! The header carries explicit offsets for every section so a reader can
//! `mmap` the file and jump straight to whichever section it needs, without
//! scanning. This matches the NSE spec (magic `"NSE1"`, fixed header, sparse
//! micro-experts, MIPS tree at the tail).

use std::fs::{File, OpenOptions};
use std::io::{Seek, SeekFrom, Write};
use std::path::Path;

use crate::error::{NseError, NseResult};

/// File magic, ASCII `"NSE1"`.
pub const NSE_MAGIC: [u8; 4] = *b"NSE1";

/// Current on-disk format version. Bump when the header layout changes.
pub const NSE_VERSION: u32 = 1;

/// Fixed size of [`NSEFileHeader`] on disk, in bytes.
pub const HEADER_SIZE: u64 = 100;

/// Top-level `.nse` file header.
///
/// The first six fields mirror the NSE technical specification exactly
/// (`magic`, `total_params`, `num_layers`, `dense_core_size`, `codebook_size`,
/// `index_tree_offset`). The remaining fields are navigation helpers that let
/// an `mmap`-based reader locate every section by offset without scanning.
#[derive(Debug, Clone, PartialEq)]
pub struct NSEFileHeader {
    /// Magic bytes, must equal [`NSE_MAGIC`].
    pub magic: [u8; 4],
    /// Format version, must equal [`NSE_VERSION`].
    pub version: u32,
    /// Total parameter count of the source dense model (e.g. 8_000_000_000).
    pub total_params: u64,
    /// Number of weight layers transmuted.
    pub num_layers: u32,
    /// Number of micro-experts stored in the file.
    pub num_micro_experts: u32,
    /// Dimensionality of each micro-expert centroid vector.
    pub centroid_dim: u32,
    /// Size of the dense outlier core section, in bytes.
    pub dense_core_size: u32,
    /// Size of the shared PQ codebook section, in bytes.
    pub codebook_size: u32,
    /// Size of the packed micro-expert data section, in bytes.
    pub micro_expert_data_size: u64,
    /// Byte offset of the dense core section.
    pub dense_core_offset: u64,
    /// Byte offset of the codebook section.
    pub codebook_offset: u64,
    /// Byte offset of the micro-expert metadata array.
    pub micro_expert_meta_offset: u64,
    /// Byte offset of the packed micro-expert data.
    pub micro_expert_data_offset: u64,
    /// Byte offset of the MIPS index tree (spec field).
    pub index_tree_offset: u64,
    /// Reserved for future use, currently zero.
    pub reserved: [u8; 16],
}

impl NSEFileHeader {
    /// Build a header pre-filled with the magic and version, all offsets and
    /// sizes zero. Callers fill in the real numbers while writing.
    pub fn new(total_params: u64, num_layers: u32) -> Self {
        Self {
            magic: NSE_MAGIC,
            version: NSE_VERSION,
            total_params,
            num_layers,
            num_micro_experts: 0,
            centroid_dim: 0,
            dense_core_size: 0,
            codebook_size: 0,
            micro_expert_data_size: 0,
            dense_core_offset: HEADER_SIZE,
            codebook_offset: HEADER_SIZE,
            micro_expert_meta_offset: HEADER_SIZE,
            micro_expert_data_offset: HEADER_SIZE,
            index_tree_offset: HEADER_SIZE,
            reserved: [0u8; 16],
        }
    }

    /// Encode the header to a 96-byte little-endian buffer.
    pub fn encode(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(HEADER_SIZE as usize);
        buf.extend_from_slice(&self.magic);
        buf.extend_from_slice(&self.version.to_le_bytes());
        buf.extend_from_slice(&self.total_params.to_le_bytes());
        buf.extend_from_slice(&self.num_layers.to_le_bytes());
        buf.extend_from_slice(&self.num_micro_experts.to_le_bytes());
        buf.extend_from_slice(&self.centroid_dim.to_le_bytes());
        buf.extend_from_slice(&self.dense_core_size.to_le_bytes());
        buf.extend_from_slice(&self.codebook_size.to_le_bytes());
        buf.extend_from_slice(&self.micro_expert_data_size.to_le_bytes());
        buf.extend_from_slice(&self.dense_core_offset.to_le_bytes());
        buf.extend_from_slice(&self.codebook_offset.to_le_bytes());
        buf.extend_from_slice(&self.micro_expert_meta_offset.to_le_bytes());
        buf.extend_from_slice(&self.micro_expert_data_offset.to_le_bytes());
        buf.extend_from_slice(&self.index_tree_offset.to_le_bytes());
        buf.extend_from_slice(&self.reserved);
        // Pad to fixed size if anything was miscounted.
        while (buf.len() as u64) < HEADER_SIZE {
            buf.push(0);
        }
        debug_assert_eq!(buf.len() as u64, HEADER_SIZE);
        buf
    }

    /// Decode a header from a byte slice of at least [`HEADER_SIZE`] bytes.
    pub fn decode(buf: &[u8]) -> NseResult<Self> {
        if (buf.len() as u64) < HEADER_SIZE {
            return Err(NseError::InvalidFile(format!(
                "header too short: {} < {HEADER_SIZE}",
                buf.len()
            )));
        }
        let magic = [buf[0], buf[1], buf[2], buf[3]];
        if magic != NSE_MAGIC {
            return Err(NseError::BadMagic {
                expected: NSE_MAGIC,
                got: magic,
            });
        }
        let mut rd = Reader::new(&buf[4..]);
        let version = rd.read_u32();
        if version != NSE_VERSION {
            return Err(NseError::UnsupportedVersion(version));
        }
        let total_params = rd.read_u64();
        let num_layers = rd.read_u32();
        let num_micro_experts = rd.read_u32();
        let centroid_dim = rd.read_u32();
        let dense_core_size = rd.read_u32();
        let codebook_size = rd.read_u32();
        let micro_expert_data_size = rd.read_u64();
        let dense_core_offset = rd.read_u64();
        let codebook_offset = rd.read_u64();
        let micro_expert_meta_offset = rd.read_u64();
        let micro_expert_data_offset = rd.read_u64();
        let index_tree_offset = rd.read_u64();
        let reserved = {
            let mut r = [0u8; 16];
            r.copy_from_slice(rd.take(16));
            r
        };
        Ok(Self {
            magic,
            version,
            total_params,
            num_layers,
            num_micro_experts,
            centroid_dim,
            dense_core_size,
            codebook_size,
            micro_expert_data_size,
            dense_core_offset,
            codebook_offset,
            micro_expert_meta_offset,
            micro_expert_data_offset,
            index_tree_offset,
            reserved,
        })
    }
}

/// Fixed part of a micro-expert descriptor. The variable-length centroid
/// vector follows immediately after on disk (length `header.centroid_dim`).
#[derive(Debug, Clone, PartialEq)]
pub struct MicroExpertMeta {
    pub expert_id: u32,
    pub num_channels: u32,
    /// Byte offset of this expert's packed data within the micro-expert data
    /// section.
    pub data_offset: u64,
}

/// On-disk size of the fixed part of [`MicroExpertMeta`].
pub const ME_META_FIXED_SIZE: u64 = 16;

impl MicroExpertMeta {
    /// Total on-disk size of this expert's metadata, including its centroid.
    pub fn on_disk_size(centroid_dim: u32) -> u64 {
        ME_META_FIXED_SIZE + (centroid_dim as u64) * 4
    }

    /// Encode the fixed part plus the centroid floats.
    pub fn encode(&self, centroid: &[f32]) -> Vec<u8> {
        let mut buf =
            Vec::with_capacity(ME_META_FIXED_SIZE as usize + centroid.len() * 4);
        buf.extend_from_slice(&self.expert_id.to_le_bytes());
        buf.extend_from_slice(&self.num_channels.to_le_bytes());
        buf.extend_from_slice(&self.data_offset.to_le_bytes());
        for &v in centroid {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        buf
    }

    /// Decode one expert's metadata + centroid from a byte slice, returning
    /// the parsed value and the number of bytes consumed.
    pub fn decode(buf: &[u8], centroid_dim: u32) -> NseResult<(Self, usize)> {
        let need = ME_META_FIXED_SIZE as usize + centroid_dim as usize * 4;
        if buf.len() < need {
            return Err(NseError::InvalidFile(format!(
                "micro-expert meta too short: {} < {need}",
                buf.len()
            )));
        }
        let mut rd = Reader::new(buf);
        let expert_id = rd.read_u32();
        let num_channels = rd.read_u32();
        let data_offset = rd.read_u64();
        let _centroid: Vec<f32> = (0..centroid_dim).map(|_| rd.read_f32()).collect();
        Ok((Self { expert_id, num_channels, data_offset }, need))
    }
}

/// Incremental little-endian reader over a byte slice.
struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    fn take(&mut self, n: usize) -> &'a [u8] {
        let s = &self.buf[self.pos..self.pos + n];
        self.pos += n;
        s
    }

    fn read_u32(&mut self) -> u32 {
        let mut b = [0u8; 4];
        b.copy_from_slice(self.take(4));
        u32::from_le_bytes(b)
    }

    fn read_u64(&mut self) -> u64 {
        let mut b = [0u8; 8];
        b.copy_from_slice(self.take(8));
        u64::from_le_bytes(b)
    }

    fn read_f32(&mut self) -> f32 {
        let mut b = [0u8; 4];
        b.copy_from_slice(self.take(4));
        f32::from_le_bytes(b)
    }
}

/// Write a complete `.nse` file from its parts.
///
/// Sections that are empty (zero-length) are still recorded with an offset
/// pointing to where the next section begins.
pub fn write_nse_file(
    path: impl AsRef<Path>,
    header: &NSEFileHeader,
    dense_core: &[u8],
    codebook: &[u8],
    micro_experts: &[(MicroExpertMeta, Vec<f32>)],
    micro_expert_data: &[u8],
    index_tree: &[u8],
) -> NseResult<()> {
    if micro_experts.iter().any(|(_, c)| c.len() as u32 != header.centroid_dim) {
        return Err(NseError::InvalidFile(
            "micro-expert centroid dim mismatch".into(),
        ));
    }

    let mut f = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(path.as_ref())?;

    // Reserve the header slot; we rewrite it once offsets are known.
    f.seek(SeekFrom::Start(HEADER_SIZE))?;

    let dense_core_offset = HEADER_SIZE;
    f.write_all(dense_core)?;
    let codebook_offset = dense_core_offset + dense_core.len() as u64;
    f.write_all(codebook)?;

    let micro_expert_meta_offset = codebook_offset + codebook.len() as u64;
    for (meta, centroid) in micro_experts {
        f.write_all(&meta.encode(centroid))?;
    }

    let micro_expert_data_offset =
        micro_expert_meta_offset
            + micro_experts.iter().map(|(m, c)| m.encode(c).len() as u64).sum::<u64>();
    f.write_all(micro_expert_data)?;

    let index_tree_offset =
        micro_expert_data_offset + micro_expert_data.len() as u64;
    f.write_all(index_tree)?;

    // Final header with real offsets/sizes.
    let final_header = NSEFileHeader {
        num_micro_experts: micro_experts.len() as u32,
        dense_core_size: dense_core.len() as u32,
        codebook_size: codebook.len() as u32,
        micro_expert_data_size: micro_expert_data.len() as u64,
        dense_core_offset,
        codebook_offset,
        micro_expert_meta_offset,
        micro_expert_data_offset,
        index_tree_offset,
        ..*header
    };
    f.seek(SeekFrom::Start(0))?;
    f.write_all(&final_header.encode())?;
    f.flush()?;
    Ok(())
}

/// In-memory view of an `.nse` file: the parsed header plus raw section bytes
/// borrowed from an `mmap`.
pub struct NseFileView<'a> {
    pub header: NSEFileHeader,
    pub dense_core: &'a [u8],
    pub codebook: &'a [u8],
    pub micro_expert_meta: &'a [u8],
    pub micro_expert_data: &'a [u8],
    pub index_tree: &'a [u8],
}

impl<'a> NseFileView<'a> {
    /// Slice out the sections from a fully-loaded byte buffer using the
    /// header offsets. Used by both the mmap and the `read_to_vec` paths.
    pub fn from_bytes(buf: &'a [u8], header: NSEFileHeader) -> NseResult<Self> {
        let h = &header;
        let dense_core = slice_range(buf, h.dense_core_offset, h.dense_core_size as u64)?;
        let codebook = slice_range(buf, h.codebook_offset, h.codebook_size as u64)?;
        let me_meta_end = h.micro_expert_data_offset;
        let micro_expert_meta = slice_range(buf, h.micro_expert_meta_offset, me_meta_end - h.micro_expert_meta_offset)?;
        let micro_expert_data = slice_range(
            buf,
            h.micro_expert_data_offset,
            h.micro_expert_data_size,
        )?;
        let index_tree = slice_range(buf, h.index_tree_offset, buf.len() as u64 - h.index_tree_offset)?;
        Ok(Self {
            header,
            dense_core,
            codebook,
            micro_expert_meta,
            micro_expert_data,
            index_tree,
        })
    }

    /// Parse every micro-expert descriptor out of the metadata section.
    pub fn micro_experts(&self) -> NseResult<Vec<(MicroExpertMeta, Vec<f32>)>> {
        let mut out = Vec::with_capacity(self.header.num_micro_experts as usize);
        let mut buf = self.micro_expert_meta;
        for _ in 0..self.header.num_micro_experts {
            let (meta, used) = MicroExpertMeta::decode(buf, self.header.centroid_dim)?;
            // Re-decode the centroid (decode currently discards it).
            let mut rd = Reader::new(buf);
            rd.pos = ME_META_FIXED_SIZE as usize;
            let centroid: Vec<f32> = (0..self.header.centroid_dim)
                .map(|_| rd.read_f32())
                .collect();
            out.push((meta, centroid));
            buf = &buf[used..];
        }
        Ok(out)
    }
}

fn slice_range(buf: &[u8], offset: u64, len: u64) -> NseResult<&[u8]> {
    let start = offset as usize;
    let end = start + len as usize;
    if end > buf.len() {
        return Err(NseError::InvalidFile(format!(
            "section out of bounds: [{start}, {end}) > buf len {}",
            buf.len()
        )));
    }
    Ok(&buf[start..end])
}

/// Read an `.nse` file fully into memory and return a view over the owned
/// buffer. The buffer must outlive the view; pair with [`NseFileView::from_bytes`].
pub fn read_nse_file(path: impl AsRef<Path>) -> NseResult<(Vec<u8>, NseFileView<'static>)> {
    let buf = std::fs::read(path.as_ref())?;
    let header = NSEFileHeader::decode(&buf)?;
    // SAFETY: the returned view borrows from `buf` for `'static`, which is a
    // lie — the caller must keep `buf` alive. We expose this as a convenience
    // only within the read-all-then-use pattern. Prefer [`open_nse_mmap`].
    let view = NseFileView::from_bytes(
        unsafe { std::mem::transmute::<&[u8], &'static [u8]>(buf.as_slice()) },
        header,
    )?;
    Ok((buf, view))
}

/// Memory-map an `.nse` file read-only and return a view borrowing the mapping.
pub fn open_nse_mmap(path: impl AsRef<Path>) -> NseResult<NseFileView<'static>> {
    let file = File::open(path.as_ref())?;
    let mmap = unsafe { memmap2::Mmap::map(&file)? };
    // Leak the mapping so the view can borrow it for 'static. The mapping is
    // kept alive for the process; for the POC this is acceptable.
    let static_buf: &'static [u8] = unsafe {
        std::mem::transmute::<&[u8], &'static [u8]>(mmap.as_ref())
    };
    std::mem::forget(mmap);
    let header = NSEFileHeader::decode(static_buf)?;
    NseFileView::from_bytes(static_buf, header)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile;

    #[test]
    fn header_roundtrip() {
        let h = NSEFileHeader::new(123_456_789, 32);
        let enc = h.encode();
        assert_eq!(enc.len() as u64, HEADER_SIZE);
        let dec = NSEFileHeader::decode(&enc).unwrap();
        assert_eq!(h, dec);
    }

    #[test]
    fn bad_magic_rejected() {
        let mut h = NSEFileHeader::new(0, 0);
        h.magic = *b"NSE2";
        let enc = h.encode();
        assert!(matches!(
            NSEFileHeader::decode(&enc),
            Err(NseError::BadMagic { .. })
        ));
    }

    #[test]
    fn micro_expert_roundtrip() {
        let meta = MicroExpertMeta {
            expert_id: 7,
            num_channels: 64,
            data_offset: 0x1000,
        };
        let centroid = vec![0.1, -0.2, 0.3, 0.0];
        let enc = meta.encode(&centroid);
        let (dec, used) = MicroExpertMeta::decode(&enc, 4).unwrap();
        assert_eq!(used, enc.len());
        assert_eq!(meta, dec);
    }

    #[test]
    fn full_file_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("m.nse");

        let header = NSEFileHeader::new(1_000_000, 4);
        let header = NSEFileHeader { centroid_dim: 2, ..header };
        let dense_core = vec![0xAAu8; 32];
        let codebook = vec![0u8; 0];
        let me1 = (
            MicroExpertMeta { expert_id: 0, num_channels: 8, data_offset: 0 },
            vec![0.5, -0.5],
        );
        let me2 = (
            MicroExpertMeta { expert_id: 1, num_channels: 8, data_offset: 16 },
            vec![0.25, 0.75],
        );
        let me_data = vec![0b01_10_00_01u8; 16];
        let index_tree = vec![0xCCu8; 8];

        write_nse_file(
            &path,
            &header,
            &dense_core,
            &codebook,
            &[me1.clone(), me2.clone()],
            &me_data,
            &index_tree,
        )
        .unwrap();

        let (buf, view) = read_nse_file(&path).unwrap();
        assert_eq!(view.header.num_micro_experts, 2);
        assert_eq!(view.header.centroid_dim, 2);
        assert_eq!(view.dense_core, &dense_core[..]);
        assert_eq!(view.micro_expert_data, &me_data[..]);
        assert_eq!(view.index_tree, &index_tree[..]);

        let experts = view.micro_experts().unwrap();
        assert_eq!(experts.len(), 2);
        assert_eq!(experts[0].0, me1.0);
        assert_eq!(experts[0].1, me1.1);
        assert_eq!(experts[1].0, me2.0);
        // keep buf alive
        drop(buf);
    }
}
