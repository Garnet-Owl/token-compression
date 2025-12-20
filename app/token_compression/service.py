from typing import Optional, Dict

from app.shared.logger import get_logger
from app.token_compression.compressor import Compressor
from app.token_compression.config import CompressionConfig
from app.token_compression.decompressor import Decompressor

logger = get_logger(__name__)


class CompressionService:
    """Facade for compression and decompression services."""

    def __init__(self, config: Optional[CompressionConfig] = None):
        self.config = config or CompressionConfig()
        self.compressor = Compressor(self.config)
        self.decompressor = Decompressor()

    def compress(self, text: str) -> str:
        return self.compressor.compress(text)

    def decompress(self, compressed_text: str) -> str:
        return self.decompressor.decompress(compressed_text)

    def get_stats(self) -> Dict:
        return self.compressor.get_stats()

    @staticmethod
    def verify_roundtrip(original: str, decompressed: str) -> bool:
        orig_norm = " ".join(original.split())
        decomp_norm = " ".join(decompressed.split())
        return orig_norm == decomp_norm


if __name__ == "__main__":

    def main():
        tests = [
            (
                "Highly Repetitive",
                """
            The quick brown fox jumps over the lazy dog. The quick brown fox
            jumps over the lazy dog again. And again, the quick brown fox
            jumps over the lazy dog. The lazy dog never moves because the
            lazy dog is lazy. The quick brown fox keeps jumping over the
            lazy dog because that is what the quick brown fox does.
            """,
            ),
            (
                "Technical Text",
                """
            The database connection pool manages database connections. When a
            service needs a database connection, it requests one from the
            database connection pool. The database connection pool returns an
            available database connection. After the service is done with the
            database connection, it returns the database connection to the
            database connection pool.
            """,
            ),
            (
                "Natural Language",
                """
            what points are missing in this proposal based on this call:
            Topic description
            Expected Outcome:
            Project results are expected to contribute to all the following expected outcomes:

            Knowledge of high-risk locations along the road network becoming available, before crashes actually occur, enabling road authorities to deploy appropriate countermeasures proactively;
            Predictive identification of safety-critical situations based on data from multiple sources and enabling real-time interventions to avoid crashes;
            Determination of the optimal sample size to allow for reliable real-time crash occurrence prediction;
            Enhanced monitoring of traffic flows and incorporation of traffic flow variations and patterns in real-time crash prediction, which will also lead to more effective traffic management by foreseeing unexpected or disruptive events.
            Scope:
            One of the principles of the Safe System Approach is to turn from mainly re-active to pro-active management of road safety, i.e. not to derive needs for intervention primarily from crash investigations, but to intervene before serious crashes happen. The ubiquitous gathering of ever-growing amounts of data and their processing in the digital transport system support this idea providing valuable information on traffic situations and events. Potential data sources include amongst others: smart phones, wearables, connected vehicles, drones, road-side sensors (e.g. camera, radar), etc. Progress in computing power, in the accuracy of location services and in video analytics are further enablers in the processing and analysis of such data in order to identify safety-critical situations or conflicts based on surrogate safety metrics.

            In terms of crash prediction modelling artificial intelligence has the potential to identify the underlying risk and the complex relationships between large and diverse datasets which in turn could lead to the identification of crash contributing factors and their interrelations. The identification of these risk factors may then allow predicting safety-critical situations at quantifiable risk levels and guide the proactive implementation of crash avoidance measures, as proposed amongst others by the International Transport Forum at the Organisation for Economic Co-operation and Development (OECD). Ideally, interventions would be feasible in real-time and increase the safety of all road users.

            Proposals should address all the following aspects:

            Development of an artificial intelligence (AI)-enabled digital twin of traffic and infrastructure. This would integrate historical, current, and forecast data, including crowdsourcing and infrastructure sensors, infrastructure topology and condition, along with environmental (e.g. local weather and visibility) and road and traffic conditions. Such a digital twin can allow monitoring and preventively optimising both safety and traffic flow, equally addressing congestion and resilience issues. Results from existing projects like OMICRON[1] could be considered. The proposals should also explore the possibility and usefulness of other type of data such as sociodemographic and economic data, behavioural driving data, data from security cameras, among others that could be provided by third parties (tourism, planned events, demand, etc.);
            Analyse in detail the technical challenges associated with the acquisition and use of adequate and reliable big data from multiple sensors in the road transport system, as well as the process of combining these datasets in ways that are meaningful for proactive road safety analysis;
            Develop methods and tools to predict safety-critical traffic situations at quantifiable risk levels based on real-time and historical data;
            Account for biases in the datasets and ensure that the developed AI-based models or algorithms are bias-free, so that the safety of all road users will be improved effectively in a fair, non-discriminatory way;
            Analyse in detail also the non-technical challenges associated with this approach and the inherent need to collect and share large amounts of data that can be used to identify and quantify road safety-related risk factors. Ethical, legal and economic issues should be considered and concepts be developed to overcome these challenges in terms of privacy concerns, questions of data ownership, organisational barriers etc;
            Analyse what real-time countermeasures can be taken to reduce instantaneous risk levels for all road users complementary to existing Intelligent Transport Systems (ITS) services;
            Demonstrate the feasibility of such risk predictions and targeted interventions;
            Build consensus among relevant stakeholders on possible routes for deployment in coordination with other ITS services.
            Particular attention should be dedicated on establishing interoperability standards for data sharing, through the implementation of the FAIR (Findable, Accessible, Interoperable, and Reusable) data principles and leveraging on already adopted practices especially those in the relevant Common European data spaces.

            Ways to leverage valuable complementary data, e.g. metadata from crash databases, should also be explored, as well as links to initiatives for European data spaces.

            Research is expected to develop recommendations for updates to relevant standards and legal frameworks. International cooperation is advised, in particular with projects or partners from the US, Japan, Singapore and Australia. Knowledge and experience from other modes where similar approaches are followed in much more controlled environments should be leveraged.
            """,
            ),
        ]

        service = CompressionService()

        logger.info("=" * 70)
        logger.info("LLM Text Compression System - Service Demo")
        logger.info("=" * 70)

        all_passed = True

        for name, text in tests:
            logger.info(f"{'─' * 70}")
            logger.info(f"Test: {name}")

            compressed = service.compress(text)
            logger.info(f"Compressed text: {compressed}")
            decompressed = service.decompress(compressed)
            stats = service.get_stats()
            passed = service.verify_roundtrip(text, decompressed)
            all_passed &= passed

            logger.info(f"Original: {stats['original_chars']} chars, ~{stats['original_tokens']} tokens")
            logger.info(f"Compressed: {stats['compressed_chars']} chars, ~{stats['compressed_tokens']} tokens")
            logger.info(f"Ratio: {stats['compression_ratio']}x, Saved: {stats['space_saved_pct']}%")
            logger.info(f"Token reduction: {stats['token_reduction_pct']}%")
            logger.info(f"Dict subs: {stats['dict_subs']}, LZ77 refs: {stats['lz77_refs']}")
            logger.info(f"Roundtrip: {'✓ PASS' if passed else '✗ FAIL'}")

        logger.info("=" * 70)
        logger.info(f"All tests passed: {'✓ YES' if all_passed else '✗ NO'}")

    main()
