from fastapi import FastAPI
from pydantic import BaseModel
from gradio_client import Client
from fastapi.middleware.cors import CORSMiddleware
from Bio import Entrez, SeqIO

client = Client("https://rajaatif786-vhbert.hf.space")


app = FastAPI(title="Proxy API for VirTransformer")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development/testing. Use exact domain in prod.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
class DNAInput(BaseModel):
    seq: str

@app.post("/predict_sequence")
def predict_sequence(input: DNAInput):
    try:
        result = client.predict(
            input.seq,
            api_name="//predict_dna"  # this must match your Hugging Face `api_name`
        )
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}

Entrez.email = "rajaatif78600000@gmail.com"  # Required by NCBI

# Input model
class DNAInput(BaseModel):
    seq: str  # Will be an accession ID in this case

# Helper: Fetch DNA sequence from NCBI
def fetch_sequence_from_accession(accession_id):
    try:
        with Entrez.efetch(db="nucleotide", id=accession_id, rettype="fasta", retmode="text") as handle:
            record = SeqIO.read(handle, "fasta")
            return str(record.seq)
    except Exception as e:
        raise RuntimeError(f"Error fetching sequence: {e}")


# Helper: Fetch GFF3-like data from GenBank
def fetch_gff3_from_accession(accession_id):
    gff_data = "##gff-version 3\n"
    try:
        with Entrez.efetch(db="nucleotide", id=accession_id, rettype="gb", retmode="text") as handle:
            record = SeqIO.read(handle, "genbank")
            seqid = record.id

            for idx, feature in enumerate(record.features):
                if feature.type in ["gene", "CDS", "mRNA"]:
                    start = int(feature.location.start) + 1  # GFF is 1-based
                    end = int(feature.location.end)
                    strand = "+" if feature.location.strand == 1 else "-" if feature.location.strand == -1 else "."
                    score = "."  # You can use "." or a feature.qualifier value
                    attributes = f"ID={feature.type}{idx+1}"

                    # Optionally include gene name
                    if "gene" in feature.qualifiers:
                        attributes += f";Name={feature.qualifiers['gene'][0]}"
                    elif "product" in feature.qualifiers:
                        attributes += f";Name={feature.qualifiers['product'][0]}"

                    gff_data += f"{seqid}\tNCBI\t{feature.type}\t{start}\t{end}\t{score}\t{strand}\t.\t{attributes}\n"
        return gff_data
    except Exception as e:
        raise RuntimeError(f"Error fetching annotations: {e}")
# Endpoint: Accession to DNA → Prediction
@app.post("/predict_accession")
def predict_accession(input: DNAInput):
    try:
        accession_id = input.seq.strip()
        dna_sequence = fetch_sequence_from_accession(accession_id)
        gff_data = fetch_gff3_from_accession(accession_id)

        # Call your Hugging Face model API
        result = client.predict(
            dna_sequence,
            api_name="//predict_dna"  # make sure your space exposes this endpoint
        )
        
        result['annotations_gff'] = gff_data
        return {"accession": accession_id, "sequence": dna_sequence, "result": result}

    except Exception as e:
        return {"error": str(e)}
