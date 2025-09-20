use anyhow::{anyhow, Result};
use clap::{Parser, Subcommand};
use hex::FromHex;
use rand::rngs::OsRng;
use sha2::{Digest, Sha256};
use schnorrkel::{
    signing_context, ExpansionMode, Keypair, MiniSecretKey, PublicKey, Signature,
};

/// Domain separation for attestation messages.
const ATTEST_CTX: &[u8] = b"sr25519-hard-child-attest:v1";

/// CLI
#[derive(Parser, Debug)]
#[command(author, version, about)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand, Debug)]
enum Cmd {
    /// Generate a random parent key and derive a hard child at index, then attest.
    Demo {
        /// Derivation index (u32)
        #[arg(long, default_value_t = 0u32)]
        index: u32,
    },
    /// Use a provided 32-byte seed (hex) as the parent mini secret, derive child, and attest.
    Derive {
        /// 32-byte hex mini secret (NOT the raw private scalar; treat carefully!)
        #[arg(long)]
        parent_mini_hex: String,
        /// Derivation index (u32)
        #[arg(long)]
        index: u32,
    },
    /// Verify an attestation from the parent that (parent_pk -> child_pk, index, cc) is valid.
    Verify {
        /// Parent public key (hex, 32 bytes)
        #[arg(long)]
        parent_pk_hex: String,
        /// Child public key (hex, 32 bytes)
        #[arg(long)]
        child_pk_hex: String,
        /// Index (u32)
        #[arg(long)]
        index: u32,
        /// Chain code (hex, 32 bytes) that was used for hard derivation
        #[arg(long)]
        cc_hex: String,
        /// Signature (hex, 64 bytes)
        #[arg(long)]
        sig_hex: String,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Demo { index } => {
            // 1) Make a random parent mini secret
            let parent_mini = MiniSecretKey::generate_with(OsRng);
            let (bundle, cc) = hard_derive_from_mini(&parent_mini, index)?;

            // 2) Parent attests child linkage
            let sig = parent_attest(&bundle.parent_kp, &bundle.child_pk, index, &cc)?;

            print_bundle(&bundle, &cc, &sig);
        }
        Cmd::Derive {
            parent_mini_hex,
            index,
        } => {
            let bytes: [u8; 32] = <[u8; 32]>::from_hex(parent_mini_hex.trim_start_matches("0x"))?;
            let parent_mini = MiniSecretKey::from_bytes(&bytes)
                .map_err(|_| anyhow!("Invalid MiniSecretKey bytes"))?;

            let (bundle, cc) = hard_derive_from_mini(&parent_mini, index)?;
            let sig = parent_attest(&bundle.parent_kp, &bundle.child_pk, index, &cc)?;

            print_bundle(&bundle, &cc, &sig);
        }
        Cmd::Verify {
            parent_pk_hex,
            child_pk_hex,
            index,
            cc_hex,
            sig_hex,
        } => {
            let parent_pk_bytes: [u8; 32] =
                <[u8; 32]>::from_hex(parent_pk_hex.trim_start_matches("0x"))?;
            let child_pk_bytes: [u8; 32] =
                <[u8; 32]>::from_hex(child_pk_hex.trim_start_matches("0x"))?;
            let cc_bytes: [u8; 32] = <[u8; 32]>::from_hex(cc_hex.trim_start_matches("0x"))?;
            let sig_bytes: [u8; 64] = <[u8; 64]>::from_hex(sig_hex.trim_start_matches("0x"))?;

            let parent_pk = PublicKey::from_bytes(&parent_pk_bytes)
                .map_err(|_| anyhow!("Bad parent public key"))?;
            let child_pk = PublicKey::from_bytes(&child_pk_bytes)
                .map_err(|_| anyhow!("Bad child public key"))?;
            let sig = Signature::from_bytes(&sig_bytes);

            let ok = verify_attestation(&parent_pk, &child_pk, index, &cc_bytes, &sig)?;
            println!("attestation_valid={}", ok);
        }
    }
    Ok(())
}

/// Derivation + attestation output we print.
struct DeriveBundle {
    parent_kp: Keypair,
    parent_pk_hex: String,
    child_pk_hex: String,
    index: u32,
}

/// Print the result bundle in hex form.
fn print_bundle(bundle: &DeriveBundle, cc: &[u8; 32], sig: &Signature) {
    println!("parent_pk=0x{}", hex::encode(bundle.parent_kp.public.to_bytes()));
    println!("child_pk=0x{}", bundle.child_pk_hex);
    println!("index={}", bundle.index);
    println!("cc=0x{}", hex::encode(cc));
    println!("attestation_sig=0x{}", hex::encode(sig.to_bytes()));
    println!();
    println!("# How to verify:");
    println!("sr25519-hard-derive-attest verify \\");
    println!("  --parent-pk-hex 0x{} \\", hex::encode(bundle.parent_kp.public.to_bytes()));
    println!("  --child-pk-hex  0x{} \\", bundle.child_pk_hex);
    println!("  --index {} \\", bundle.index);
    println!("  --cc-hex 0x{} \\", hex::encode(cc));
    println!("  --sig-hex 0x{}", hex::encode(sig.to_bytes()));
}

/// Compute the 32-byte chain code from your recipe: cc = sha256(sk_bytes)||index_le (and then re-hash to 32 bytes).
///
/// We:
///   1) take the parent *mini secret* 32 bytes (`msk_bytes`)
///   2) do H = sha256(msk_bytes || index_le)
///   3) reduce to 32 bytes (sha256 output already 32)
fn chain_code_from_recipe(parent_mini: &MiniSecretKey, index: u32) -> [u8; 32] {
    let msk_bytes = parent_mini.to_bytes();
    let mut hasher = Sha256::new();
    hasher.update(&msk_bytes);
    hasher.update(index.to_le_bytes());
    let h = hasher.finalize(); // 32 bytes
    let mut cc = [0u8; 32];
    cc.copy_from_slice(&h);
    cc
}

/// Perform **hard** derivation from a parent MiniSecretKey using a 32-byte chain code,
/// returning the child and also the parent/child pubkeys for convenience.
///
/// We deliberately use a hard derivation primitive from `schnorrkel` that takes a
/// chain code and does the non-publicly linkable derivation.
fn hard_derive_from_mini(parent_mini: &MiniSecretKey, index: u32) -> Result<(DeriveBundle, [u8; 32])> {
    let cc = chain_code_from_recipe(parent_mini, index);

    // Expand parent mini secret into a sr25519 Keypair.
    // We use ExpansionMode::Uniform which is standard for sr25519 in schnorrkel.
    let parent_kp = parent_mini.expand_to_keypair(ExpansionMode::Uniform);

    // schnorrkel exposes a hard-derivation method on MiniSecretKey:
    //   MiniSecretKey::hard_derive_mini_secret_key(Some(chain_code), &cc)
    //
    // It returns (MiniSecretKey, chain_code_for_next).
    let (child_mini, _next_cc) = parent_mini
        .hard_derive_mini_secret_key(Some(&cc), &[])
        .map_err(|e| anyhow!("hard_derive failed: {:?}", e))?;

    let child_kp = child_mini.expand_to_keypair(ExpansionMode::Uniform);
    let child_pk_hex = hex::encode(child_kp.public.to_bytes());

    let bundle = DeriveBundle {
        parent_kp,
        parent_pk_hex: String::new(), // not used by printing (we pull from parent_kp directly)
        child_pk_hex,
        index,
    };
    Ok((bundle, cc))
}

/// Make a parent **attestation**: parent signs a message that binds (child_pk, index, cc).
fn parent_attest(parent: &Keypair, child_pk: &PublicKey, index: u32, cc: &[u8; 32]) -> Result<Signature> {
    let msg = attestation_message(child_pk, index, cc);
    let ctx = signing_context(ATTEST_CTX);
    Ok(parent.sign(ctx.bytes(&msg)))
}

/// Verify the parent attestation on (child_pk, index, cc) with parent public key.
fn verify_attestation(parent_pk: &PublicKey, child_pk: &PublicKey, index: u32, cc: &[u8; 32], sig: &Signature) -> Result<bool> {
    let msg = attestation_message(child_pk, index, cc);
    let ctx = signing_context(ATTEST_CTX);
    Ok(parent_pk.verify(ctx.bytes(&msg), sig).is_ok())
}

/// Deterministic message to sign/verify for the attestation.
/// Layout:
///   "sr25519-hard-child" || parent_version(1 byte) || child_pk(32) || index(4 LE) || cc(32)
fn attestation_message(child_pk: &PublicKey, index: u32, cc: &[u8; 32]) -> Vec<u8> {
    let mut m = Vec::with_capacity(1 + 32 + 4 + 32 + 24);
    m.extend_from_slice(b"sr25519-hard-child");
    m.extend_from_slice(&[1u8]); // version
    m.extend_from_slice(&child_pk.to_bytes());
    m.extend_from_slice(&index.to_le_bytes());
    m.extend_from_slice(cc);
    m
}

/* =========================================================================================
   Notes

   - This sample demonstrates the *practical* way to prove linkage for a hard-derived key:
     a **parent attestation**. The verifier never learns the parent secret, but can verify
     that whoever controls the parent key authorized/binds this child for (index, cc).

   - The derivation uses schnorrkel’s **hard** derivation (unlinkable from public data).
     Because it’s hard (not soft), a verifier cannot recompute the child from `(parent_pk, cc)`
     —that’s by design.

   - If you need a purely math/cryptographic proof without any parent signature,
     you’d build a **zero-knowledge proof** that:
         1) you know `sk` such that `PK_parent = sk·G`, and
         2) `PK_child` equals the result of schnorrkel’s hard-derive on `sk` with `cc`.
     That’s a separate circuit (R1CS/PLONK/etc.) and out of scope for a tiny demo.

   - Security caveat: Your `cc = sha256(msk) || index` choice is fine for *binding* an index to a
     chain-code, but consider including a domain tag (e.g., "path:hard") in the hash preimage to
     avoid cross-protocol reuse, and never reveal the mini secret.
========================================================================================= */
